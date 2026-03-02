import numpy as np
from sklearn.metrics import auc

def eval_retrieval_metric(rank_matrix: np.ndarray, group_item_label: np.ndarray, gt_group_label: np.ndarray):
    """Compute retrieval metrics given ranklist and groundtruth labels

    Args:
        rank_matrix: matrix of shape (num_queries, top-k) denoting the IDs (in range [0..num_items-1]) of the retrieved items
        group_item_label: class labels for items, shape (num_items, )
        gt_group_label: groundtruth class labels for queries, shape (num_queries, )

    Returns:
        PRC, NN, P@10, NDCG, mAP, auc, MRR
    """
    num_query, top_k = rank_matrix.shape
    precision = np.zeros((num_query, top_k))
    recall = np.zeros((num_query, top_k))
    # transform rank matrix to 0-1 matrix denoting irrelevance/relevance
    # image_label[..., np.newaxis] is broadcasted to (num_queries, 1)
    rel_matrix = group_item_label[rank_matrix] == gt_group_label[..., np.newaxis] # (num_queries, top_k), each entry is 0 or 1
    ap = np.zeros(num_query)
    reciprocal_rank = np.zeros(num_query)

    for i in range(num_query):
        max_match = np.sum(group_item_label == gt_group_label[i]) # Get all relevant models

        
        G_sum = np.cumsum(np.int8(rel_matrix[i])) # (top_k, ) # since there is only top_k elements, the max entry can be top_k
        
        total = G_sum[-1]
        r1, p1 = np.zeros(top_k), np.zeros(top_k)
        r_points = []

        if total > 0:
            r_points = [np.where(G_sum == j)[0][0] + 1 for j in range(1, G_sum[-1] + 1)]
            r_points_int = np.array(r_points, dtype=int)

            ap[i] = np.mean(G_sum[r_points_int-1] / r_points)
            r1 = G_sum / float(max_match)
            p1 = G_sum / np.arange(1, len(G_sum)+1)

        recall[i] = r1
        precision[i] = p1

        reciprocal_rank[i] = 1.0 / r_points[0] if r_points and r_points[0] != 0 else 0

    map_ = np.mean(ap)
    mrr = np.mean(reciprocal_rank)
    nn = np.mean(rel_matrix[:, 0])
    precison_at = [np.mean(precision[:, k - 1]) for k in range(1, top_k + 1)]
    recall_at = [np.mean(recall[:, k - 1]) for k in range(1, top_k + 1)]


    logf = np.log2(1 + np.arange(1, top_k + 1))[np.newaxis, :]  # reduction factor for DCG
    dcg = np.sum(rel_matrix / logf, axis=-1)        # (num_queries, )

    idcg = np.zeros(num_query)
    for i in range(num_query):
        max_match = np.sum(group_item_label == gt_group_label[i])  # Total relevant items
        ideal_rel = np.zeros(top_k)
        ideal_rel[:min(max_match, top_k)] = 1  # Top relevant items get 1
        idcg[i] = np.sum(ideal_rel / logf)  # Compute IDCG properly


    ndcg = np.mean(dcg / idcg) 
    pre = np.mean(precision, axis=0)
    rec = np.mean(recall, axis=0)

    auc_ = auc(rec, pre)

    return {
        "NN": nn,
        "NDCG": ndcg,
        "mAP": map_,
        "auc": auc_,
        "P@": precison_at,
        "R@": recall_at,
        "MRR": mrr
    }
