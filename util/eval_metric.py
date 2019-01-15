import numpy as np


def average_precision(img_id_rank, gt_index):
    """
    :param img_id_rank: a list of database image id based on similarity to query
    :param gt_index: ground truth index to the query
    :return: average precision for the query
    """

    gt_index = set(gt_index)
    relative_score = [1 if x in gt_index else 0 for x in img_id_rank]
    num_list = np.arange(1, len(relative_score)+1, dtype=np.float)

    precision = np.cumsum(relative_score)/num_list
    recall = np.cumsum(relative_score)/len(gt_index)

    A = np.append(np.array([0.0]), recall)
    B = np.append(np.array([1.0]), precision[0:-1]) + precision

    ap = np.dot(np.diff(A), B)/2.0
    # convert ap from numpy.float64 type to the python internal float type
    ap = float(ap)

    return ap


def recall_at_k(position, img_id_rank, gt_index):
    """
    :param position: the k nearest negighbor to search, position is a list
    :param img_id_rank: rank of test set images
    :param gt_index: ground truth image index
    :return: a list of score for each k: 1 if k nearest neighbor contains
            true-positive else 0
    """
    score = [0]*len(position)
    gt_index = set(gt_index)

    for i, k in enumerate(position):

        if not set(img_id_rank[:k]).isdisjoint(gt_index):

            for j in range(i, len(position)):
                score[j] = 1
            break

    return score
