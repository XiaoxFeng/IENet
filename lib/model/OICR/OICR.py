from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.net as net_utils
from core.config import cfg

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



def OICR(boxes, cls_prob, im_labels, cls_prob_new, rot_inds):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    rot_inds = rot_inds.data[0].cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, rot_inds)

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
        rot_labels = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy(), rot_inds)

    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy(),
            'rot_labels': rot_labels.reshape(1, -1).astype(np.int64).copy()}

def _get_highest_score_proposals(boxes, cls_prob, im_labels, rot_inds):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    rot_labels = np.zeros((0, 1), dtype=np.float32)

    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)
            #max_box = np.zeros((0, 4), dtype=np.float32)
            #max_box = np.vstack((max_box, boxes[max_index, :].reshape(1, -1)))
            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            rot_labels = np.vstack((rot_labels,
                                   rot_inds * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores,
                 'rot_labels': rot_labels}

    return proposals


def _get_highest_score_proposals_rot(boxes, cls_prob, im_labels, rot_score, rot_inds):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    rot_labels = np.zeros((0, 1), dtype=np.float32)

    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]
            cls_prob_rot = cls_prob_tmp + rot_score[idxs] * 0.6
            max_index = np.argmax(cls_prob_rot)
            gt_boxes = np.vstack((gt_boxes, boxes_tmp[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            rot_labels = np.vstack((rot_labels,
                                    rot_inds * np.ones((1, 1), dtype=np.float32)))
            cls_prob[idxs[max_index], :] = 0
            rot_score[idxs[max_index]] = 0

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores,
                 'rot_labels': rot_labels}

    return proposals

def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER,
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))

    return (overlaps > iou_threshold).astype(np.float32)


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.MODEL.BBOX_REG_WEIGHTS)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES + 1

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def get_proposal_clusters(all_rois, proposals, im_labels, rot_inds):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    rot_ids = proposals['rot_labels']
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    rot_labels = rot_ids[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    rot_labels[bg_inds] = 4

    if cfg.MODEL.WITH_FRCNN:
        bbox_targets = _compute_targets(all_rois, gt_boxes[gt_assignment, :],
            labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype) \
            * cls_loss_weights.reshape(-1, 1)
    else:
        bbox_targets, bbox_inside_weights, bbox_outside_weights = np.array([0]), np.array([0]), np.array([0])

    gt_assignment[bg_inds] = -1

    return labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights, rot_labels

class OICRLosses(nn.Module):
    def __init__(self):
        super(OICRLosses, self).__init__()

    def forward(self, prob, labels, cls_loss_weights, gt_assignments, eps = 1e-6):
        loss = torch.log(prob + eps)[range(prob.size(0)), labels]
        loss *= -cls_loss_weights
        ret = loss.mean()
        return ret
