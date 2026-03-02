import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset, FashionIQDataset, AoDaiDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def extract_aodai_index_features(dataset, clip_model):
    """
    Extracts features for all unique real images and all sketches.
    """
    # Use a dictionary to store unique images to avoid duplicates in the gallery
    unique_images_map = {} # {image_name: feature_tensor}
    all_sketches_features = []
    all_sketches_names = []

    # Standard loader
    loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4)

    print(f"Extracting AoDai features...")
    for sketches, sketches_names,images_list, names_list in tqdm(loader):
        # sketches: [B, 3, 224, 224]
        # images_list: List of length B, where each element is a List of 3 image tensors
        
        with torch.no_grad():
            # 1. Process Sketches
            sk_feats = clip_model.encode_image(sketches.to(device))
            all_sketches_features.append(sk_feats.cpu())
            # Assuming you store sketch names in your dataset to map them later
            all_sketches_names.extend(sketches_names)

            # 2. Process Real Images
            for i, names in enumerate(names_list):
                # names is a list of 3 strings
                for j, img_name in enumerate(names):
                    if img_name not in unique_images_map:
                        img_tensor = images_list[i][j].unsqueeze(0).to(device)
                        img_feat = clip_model.encode_image(img_tensor)
                        unique_images_map[img_name] = img_feat.cpu()

    # Consolidate Gallery
    gallery_names = list(unique_images_map.keys())
    gallery_features = torch.cat(list(unique_images_map.values()), dim=0)
    
    # Consolidate Sketches
    sketch_features = torch.cat(all_sketches_features, dim=0)
    print(f"Extracted {gallery_features.shape[0]} unique gallery features and {sketch_features.shape[0]} sketch features")
    print(f"Extracted {len(gallery_names)} unique gallery names and {len(all_sketches_names)} sketch names")
    print(f"Sample gallery names: {gallery_names[:5]}")
    print(f"Sample sketch names: {all_sketches_names[:5]}")

    return sketch_features, all_sketches_names, gallery_features, gallery_names

def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset, AoDaiDataset], clip_model: CLIP) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)
    # index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    # index_names = []
    images_features = []
    images_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    else:
        print(f"extracting AoDai {dataset.split} images features")
    for images, image_names in tqdm(classic_val_loader): # images is list of images in the batch
        # images = torch.stack(images).to(device, non_blocking=True)  # convert list to tensor
        with torch.no_grad():
            images_feature = clip_model.encode_image(images.to(device))
            images_features.append(images_feature.cpu())
            images_names.extend(image_names)
    all_image_feat = torch.cat(images_features, dim = 0)
    
    print(f"Extracted  {all_image_feat.shape[0]} images features")
    print(f"Extracted {len(images_names)} images names")
    return all_image_feat, images_names

def extract_sketch_features(dataset: Union[CIRRDataset, FashionIQDataset, AoDaiDataset], clip_model: CLIP) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)
    # index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    # index_names = []
    sketch_features = []
    sketch_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    else:
        print(f"extracting AoDai {dataset.split} sketch features")
    for sketch_name, sketch_images, _, _ in tqdm(classic_val_loader): # images is list of images in the batch
        with torch.no_grad():
            sketch_feature = clip_model.encode_image(sketch_images.to(device))
            sketch_features.append(sketch_feature.cpu())
            sketch_names.extend(sketch_name)
    
    all_sketch_feat = torch.cat(sketch_features, dim = 0)
    print(f"Extracted {all_sketch_feat.shape[0]} sketches features")
    print(f"Extracted {len(sketch_names)} sketch names")
    return all_sketch_feat, sketch_names


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))
