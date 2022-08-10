from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from core.config import cfg
from model.OICR.OICR import OICR, OICRLosses
from ops import RoIPool, RoIAlign
import modeling.oicr_heads as oicr_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.vgg_weights_helper as vgg_utils
import matplotlib.pyplot as plt
import random

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_MIL_Outs =oicr_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = oicr_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.EQ_loss = OICRLosses()

        self.Refine_Losses = [OICRLosses() for i in range(cfg.REFINE_TIMES)]

        if cfg.MODEL.WITH_FRCNN:
            self.FRCNN_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
            self.Cls_Loss = OICRLosses()


        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            if cfg.MODEL.CONV_BODY.split('.')[0] == 'vgg16':
                vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, data_aug, data_aug1, rois, labels, data_rot=None, rot_inds = None):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, data_aug, data_aug1, rois, labels, data_rot, rot_inds)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, data_aug, data_aug1, rois, labels, data_rot, rot_inds)

    def _forward(self, data, data_aug, data_aug1, rois, labels, data_rot, rot_inds):
        im_data = data #torch.cat((data, data_aug))
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            rois_o, rois_flip, rois_scale, rois_T, rois_FT = rois.chunk(5, dim=0)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(data).contiguous()

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if self.training:
            blob_conv_aug = self.Conv_Body(data_aug).contiguous()

            blob_conv_scale = self.Conv_Body(data_aug1).contiguous()

            blob_conv_rot = self.Conv_Body(data_rot).contiguous()

            if rot_inds[0] == 0:
                blob_conv_T = torch.rot90(self.Conv_Body(data_rot), -1, [2, 3]).contiguous()
            elif rot_inds[0] == 1:
                blob_conv_T = torch.rot90(self.Conv_Body(data_rot), -2, [2, 3]).contiguous()
            elif rot_inds[0] == 2:
                blob_conv_T = torch.rot90(self.Conv_Body(data_rot), 1, [2, 3]).contiguous()
            else:
                blob_conv_T = blob_conv_rot

            box_feat = self.Box_Head(blob_conv, rois_o)
            box_feat_aug = self.Box_Head(blob_conv_aug, rois_flip)
            box_feat_scale = self.Box_Head(blob_conv_scale, rois_scale)
            box_feat_T = self.Box_Head(blob_conv_rot, rois_T)
            box_feat_FT = self.Box_Head(blob_conv_T, rois_FT)

        else:
            box_feat = self.Box_Head(blob_conv, rois)
            box_feat_aug = box_feat
            box_feat_scale = box_feat
            box_feat_T = box_feat
            box_feat_FT = box_feat

        feat = [box_feat, box_feat_aug, box_feat_scale, box_feat_T]
        #random.shuffle(feat)
        mil_score = self.Box_MIL_Outs(box_feat, box_feat)
        refine_score, refine_score_r = self.Box_Refine_Outs(feat, box_feat_FT)
        if cfg.MODEL.WITH_FRCNN:
            cls_score, bbox_pred = self.FRCNN_Outs(box_feat)

        device = box_feat.device

        if self.training:
            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = oicr_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            # refinement loss
            boxes = rois_o.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            for i_refine, refine in enumerate(refine_score[:3]):
                if i_refine == 0:
                    IENet_output = OICR(boxes, mil_score, im_labels, refine, rot_inds)
                else:
                    IENet_output = OICR(boxes, refine_score[i_refine - 1],
                                      im_labels, refine, rot_inds)

                refine_loss = self.Refine_Losses[i_refine](
                    refine,
                    Variable(torch.from_numpy(IENet_output['labels'])).to(device),
                    Variable(torch.from_numpy(IENet_output['cls_loss_weights'])).to(device),
                    Variable(torch.from_numpy(IENet_output['gt_assignment'])).to(device))

                RI_refine_loss = self.Refine_Losses[i_refine](
                    refine_score_r[i_refine],
                    Variable(torch.from_numpy(IENet_output['labels'])).to(device),
                    Variable(torch.from_numpy(IENet_output['cls_loss_weights'])).to(device),
                    Variable(torch.from_numpy(IENet_output['gt_assignment'])).to(device))

                if i_refine == 0:
                    refine_loss *= 3.0
                    #Eq_refine_loss *= 3.0

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()
                return_dict['losses']['refine_loss_RI%d' % i_refine] = RI_refine_loss.clone()

            eq_loss = self.EQ_loss(
                    refine_score[3],
                    Variable(torch.from_numpy(IENet_output['rot_labels'])).to(device),
                    Variable(torch.from_numpy(IENet_output['cls_loss_weights'])).to(device),
                    Variable(torch.from_numpy(IENet_output['gt_assignment'])).to(device))

            return_dict['losses']['Eq_loss'] = eq_loss.clone()

            if cfg.MODEL.WITH_FRCNN:
                labels, cls_loss_weights, bbox_targets, bbox_inside_weights, \
                    bbox_outside_weights, _ = fast_rcnn_heads.get_fast_rcnn_targets(
                        boxes, refine_score, im_labels, rot_inds.data[0].cpu().numpy())

                cls_loss, bbox_loss = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred,
                    Variable(torch.from_numpy(labels)).to(device),
                    Variable(torch.from_numpy(cls_loss_weights)).to(device),
                    Variable(torch.from_numpy(bbox_targets)).to(device),
                    Variable(torch.from_numpy(bbox_inside_weights)).to(device),
                    Variable(torch.from_numpy(bbox_outside_weights)).to(device))

                return_dict['losses']['cls_loss'] = cls_loss
                return_dict['losses']['bbox_loss'] = bbox_loss

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score
            if cfg.MODEL.WITH_FRCNN:
                return_dict['cls_score'] = cls_score
                return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.
        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPool(resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoIAlign':
            xform_out = RoIAlign(
                resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
