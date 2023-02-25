import numpy as np
import os, sys, tqdm, cv2
from scipy.optimize import linear_sum_assignment


def match(pred, gt):
    # pred: (n,h,w)
    # gt: (m,h,w)
    n, h, w = pred.shape
    m, h, w = gt.shape
    pred_mask = (pred>0)
    gt_mask = (gt>0)
    # (n,w)
    union = np.logical_or(pred_mask[:,None,:,:], gt_mask[None,:,:,:]).sum(axis=(2,3))
    inter = np.logical_and(pred_mask[:,None,:,:], gt_mask[None,:,:,:]).sum(axis=(2,3))
    iou = inter / (union + 1e-8)
    # matched_idx = np.argmax(iou, axis=0) # m
    # matched_iou = np.max(iou, axis=0) # m
    # return matched_idx, matched_iou
    return iou


def mad(pred, gt):
    pred_mask = (pred>0)
    gt_mask = (gt>0)
    union_mask = np.logical_or(pred_mask, gt_mask)
    error = np.abs(pred-gt) * union_mask.astype(np.float32)
    error = error.sum(axis=(1,2)) / (union_mask.sum(axis=(1,2)) + 1.)
    score = 1 - np.minimum(error,1)
    return score


def compute_stats_per_image(pred, gt, thresh_list, func=mad):
    # matched_idx, matched_iou = match(pred, gt)
    tp_list, fp_list, fn_list = [], [], []
    if len(pred)>0 and len(gt)>0:
        iou_matrix = match(pred, gt)
        matched_i, matched_j  = linear_sum_assignment(1-iou_matrix)
        matched_iou = iou_matrix[matched_i, matched_j]
        score = func(pred[matched_i], gt[matched_j]) * matched_iou
        for thresh in thresh_list:
            tp = (score>=thresh).sum()
            fp = pred.shape[0] - tp
            fn = gt.shape[0] - tp
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
    elif len(pred) == 0:
        for thresh in thresh_list:
            tp_list.append(0)
            fp_list.append(0)
            fn_list.append(len(gt))
    else:
        for thresh in thresh_list:
            tp_list.append(0)
            fp_list.append(len(pred))
            fn_list.append(0)
    return tp_list, fp_list, fn_list


def compute_stats(pred_folder, gt_folder, thresh_list, func=mad):
    n_thresh = len(thresh_list)
    AP = [0]*n_thresh
    TP, FP, FN = [0]*n_thresh, [0]*n_thresh, [0]*n_thresh
    for item in tqdm.tqdm(sorted(os.listdir(gt_folder))):
        if not os.path.exists(os.path.join(pred_folder, item)):
            continue
        pred_images = [cv2.imread(os.path.join(pred_folder, item, im), 0)/255. for im in os.listdir(os.path.join(pred_folder, item))]
        gt_images = [cv2.imread(os.path.join(gt_folder, item, im), 0)/255. for im in os.listdir(os.path.join(gt_folder, item))]
        if len(pred_images)>0:
            pred_items = np.stack(pred_images, axis=0)
        else:
            pred_items = []
        if len(gt_images)>0:
            gt_items = np.stack(gt_images, axis=0)
        else:
            gt_items = []
        tp_list, fp_list, fn_list = compute_stats_per_image(pred_items, gt_items, thresh_list, mad)
        for i in range(0, n_thresh):
            TP[i] += tp_list[i]
            FP[i] += fp_list[i]
            FN[i] += fn_list[i]
    for i in range(0, n_thresh):
        AP[i] = 2*TP[i] / (2*TP[i] + FP[i] + FN[i] + 1e-6)
    return AP


if __name__ == "__main__":
    pred_folder = sys.argv[1]
    gt_folder = sys.argv[2]
    thresh_list = np.linspace(0.7, 0.95, 10)
    AP = compute_stats(pred_folder, gt_folder, thresh_list, mad)
    for thresh, ap in zip(thresh_list, AP):
        print("thresh={}, AP={}".format(thresh, ap))
    print("mAP={}".format(np.mean(AP)))
