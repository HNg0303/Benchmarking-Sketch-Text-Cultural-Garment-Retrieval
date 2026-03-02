import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset, AoDaiDataset
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device, extract_aodai_index_features, extract_index_features, extract_sketch_features

def generate_aodai_val_predictions(clip_model, relative_val_dataset,
                                  combining_function, index_features, index_names) -> Tuple[torch.Tensor, List[List[str]]]:
    """
    Compute AoDai predictions. 
    Note: target_names is now a List of Lists because each query has multiple targets.
    """
    print(f"Computing AoDai validation predictions")
    
    # We use a batch size of 32 similar to your FIQ implementation
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=4, pin_memory=True,
                                     shuffle=False)

    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device)
    target_names_list = []
    name_to_feat = dict(zip(index_names, index_features))

    for sketch_name, sketch_image, captions, batch_target_names in tqdm(relative_val_loader):
        # 1. Encode Text/Sketch
        # Assuming your dataset returns 'sketches' as the reference images
        text_inputs = clip.tokenize(captions, truncate=True).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Retrieve features for the input sketches (references)
            # handle single element batch if necessary as in your FIQ code
            ref_feats = torch.stack([name_to_feat[name] for name in sketch_name]).to(device)
            
            # 2. Combine Features (Image + Text)
            batch_predicted_features = combining_function(ref_feats, text_features)
            
        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        # batch_target_names is a list of lists (the 'images' field in your JSON)
        target_names_list.extend(batch_target_names)

    return predicted_features, target_names_list

def get_metrics_aodai(image_features, ref_features, target_names, answer_names):
    """
    image_features: gallery features (num_gallery, feat_dim)
    ref_features: query features (num_queries, feat_dim)
    target_names: list of gallery image paths (num_gallery,)
    answer_names: list of list of answer image paths (num_queries, num_answers_per_query) -> 3
    """
    # 1. Compute Distances and Sort
    # Using cosine distance: 1 - cosine_similarity
    distances = 1 - (ref_features @ image_features.T) # (num_queries, num_gallery)
    sorted_indices = torch.argsort(distances, dim=-1).cpu().numpy() # Sort along gallery for each query -> top k retrievals
    
    num_query = sorted_indices.shape[0]
    top_k = sorted_indices.shape[1]
    
    # 2. Prepare Relevance Matrix
    # We check if the retrieved target_name is in the answer_names for that query
    rel_matrix = np.zeros((num_query, top_k), dtype=np.int8)
    for i in range(num_query):
        # Convert list of answers to a set for O(1) lookup
        answers = set(answer_names[i])
        for j in range(top_k):
            retrieved_item = target_names[sorted_indices[i, j]]
            if retrieved_item in answers:
                rel_matrix[i, j] = 1

    # 3. Compute Metrics (Preserving your logic)
    precision = np.zeros((num_query, top_k))
    recall = np.zeros((num_query, top_k))
    ap = np.zeros(num_query)
    reciprocal_rank = np.zeros(num_query)

    for i in range(num_query):
        # In this specific task, max_match is the number of ground truth answers provided
        max_match = len(answer_names[i])
        
        # Cumulative sum of hits
        G_sum = np.cumsum(rel_matrix[i]) # cummulative sum up to rank k.
        total_hits_in_k = G_sum[-1]
        
        if total_hits_in_k > 0:
            # Find indices where hits occurred (1-based for formula consistency)
            r_points = np.where(rel_matrix[i] == 1)[0] + 1
            
            # AP calculation: average precision at each hit point
            # Precision at rank 'r' is (number of hits up to r) / r
            ap[i] = np.mean(G_sum[r_points - 1] / r_points)
            
            # Recall and Precision curves for this query
            recall[i] = G_sum / float(max_match)
            precision[i] = G_sum / np.arange(1, top_k + 1)
            
            # MRR: 1 / rank of the first hit
            reciprocal_rank[i] = 1.0 / r_points[0]

    # Aggregate Metrics
    map_ = np.mean(ap)
    mrr = np.mean(reciprocal_rank)
    # nn = np.mean(rel_matrix[:, 0]) # Nearest Neighbor (Precision@1)
    
    # # NDCG Calculation
    # logf = np.log2(1 + np.arange(1, top_k + 1))
    # dcg = np.sum(rel_matrix / logf, axis=-1)
    
    # idcg = np.zeros(num_query)
    # for i in range(num_query):
    #     max_match = len(answer_names[i])
    #     ideal_rel = np.zeros(top_k)
    #     ideal_rel[:min(max_match, top_k)] = 1
    #     idcg[i] = np.sum(ideal_rel / logf)
    
    # # Avoid division by zero if a query has no ground truth (shouldn't happen here)
    # ndcg = np.mean(dcg / (idcg + 1e-9))
    
    # # Global Precision-Recall AUC
    # avg_pre = np.mean(precision, axis=0)
    # avg_rec = np.mean(recall, axis=0)
    # auc_score = auc(avg_rec, avg_pre)

    return {
        "R@1": recall[:, 0].mean(),
        "R@5": recall[:, 4].mean() if top_k >= 5 else None,
        "R@10": recall[:, 9].mean() if top_k >= 10 else None,
        "mAP": map_,
        "P@10": np.mean(precision[:, 9]) if top_k >= 10 else None,
        "MRR": mrr
    }

def compute_aodai_val_metrics(relative_val_dataset, clip_model, index_features, index_names, images_features, images_names, combining_function) -> Tuple[float, float]:
    # 1. Get predictions
    predicted_features, target_names_list = generate_aodai_val_predictions(
        clip_model, relative_val_dataset, combining_function, index_features, index_names
    )

    # return recall_at10, recall_at50
    answer_names = []
    for i in tqdm(range(len(relative_val_dataset))):
        # dataset[i] returns (reference_img, target_img, captions, list_of_3_answers)
        # We need that list of 3 answers
        answers = relative_val_dataset[i][3]
        answer_names.append(answers)

    # 3. Compute the retrieval metrics
    # index_features: (num_gallery, dim), predicted_features: (num_queries, dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = get_metrics_aodai(
        image_features=images_features.to(device), 
        ref_features=predicted_features.to(device), 
        target_names=images_names, 
        answer_names=answer_names
    )

    return metrics

def aodai_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on AoDai validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = AoDaiDataset('val', 'classic', preprocess)
    print("Extract image features")
    image_features, image_names = extract_index_features(classic_val_dataset, clip_model)
    print(f"Sample names: {image_names[0]}")
    print("Extract sketch features")
    relative_val_dataset = AoDaiDataset('val', 'relative', preprocess)
    index_features, index_names = extract_sketch_features(relative_val_dataset, clip_model)

    return compute_aodai_val_metrics(relative_val_dataset, clip_model, index_features, index_names, image_features, image_names, combining_function)

def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(clip_model, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(clip_model: CLIP, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                   combining_function)


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    elif args.dataset.lower() == "aodai":
        metrics = aodai_val_retrieval(combining_function, clip_model, preprocess)
        for k, v in metrics.items():
            print(f"\n{k}: {v}")
        with open("../aodai_metrics_2.json", "w") as f:
            json.dump(metrics, f, indent = 4)
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
