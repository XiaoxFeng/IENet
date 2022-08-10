import numpy as np
import numpy.random as npr
import cv2
import PIL
from core.config import cfg
import utils.blob as blob_utils


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data','data_aug', 'data_aug1', 'rois', 'labels', 'data_rot', 'rot_inds']
    return blob_names


def get_minibatch(roidb, num_classes, aug):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_blob_scale, im_blob_flip, im_blob_rot, im_scales, flip_ind, rot_inds = _get_image_blob(roidb, aug)

    #assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    blobs['data'] = im_blob
    blobs['data_aug'] = im_blob_flip
    blobs['data_aug1'] = im_blob_scale
    blobs['data_rot'] = im_blob_rot
    blobs['rot_inds'] = rot_inds
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0, num_classes), dtype=np.float32)

    num_images = len(roidb)
    for im_i in range(num_images):
        labels, im_rois, flip_rois, rot_rois = _sample_rois(roidb[im_i], num_classes, rot_inds)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        if flip_ind == 0:
            rois_scale = _project_im_rois(im_rois, im_scales[im_i + 1])
        else:
            rois_scale = _project_im_rois(flip_rois, im_scales[im_i + 1])
        flip_rois = _project_im_rois(flip_rois, im_scales[im_i + 2])
        T_rois = _project_im_rois(rot_rois, im_scales[im_i + 3])
        FT_rois = _project_im_rois(im_rois, im_scales[im_i + 3])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob_this_image_flip = np.hstack((batch_ind, flip_rois))
        rois_blob_this_image_scale = np.hstack((batch_ind, rois_scale))
        rois_blob_this_image_T = np.hstack((batch_ind, T_rois))
        rois_blob_this_image_FT = np.hstack((batch_ind, FT_rois))
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            rois_blob_this_image = rois_blob_this_image[index, :]
            rois_blob_this_image_flip = rois_blob_this_image_flip[index, :]
            rois_blob_this_image_scale = rois_blob_this_image_scale[index, :]
            rois_blob_this_image_T = rois_blob_this_image_T[index, :]
            rois_blob_this_image_FT = rois_blob_this_image_FT[index, :]


        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image_flip))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image_scale))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image_T))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image_FT))

        # Add to labels blob
        labels_blob = np.vstack((labels_blob, labels))

    blobs['rois'] = rois_blob
    blobs['labels'] = labels_blob

    return blobs, True


def _sample_rois(roidb, num_classes, rot_inds):
    """Generate a random sample of RoIs"""
    labels = roidb['gt_classes']
    rois = roidb['boxes']

    if cfg.TRAIN.BATCH_SIZE_PER_IM > 0:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    else:
        batch_size = np.inf
    if batch_size < rois.shape[0]:
        rois_inds = npr.permutation(rois.shape[0])[:batch_size]
        rois = rois[rois_inds, :]
    flip_rois = rois.copy()
    width = roidb['width']
    heigt = roidb['height']
    oldx1 = flip_rois[:, 0].copy()
    oldx2 = flip_rois[:, 2].copy()
    flip_rois[:, 0] = width - oldx2 - 1
    flip_rois[:, 2] = width - oldx1 - 1
    assert (flip_rois[:, 2] >= flip_rois[:, 0]).all()
    #rois = np.vstack((rois, flip_rois))

    for i in rot_inds:
        if i == 0:
            # ------------------rotate90-------------------
            r90_rois = rois.copy()
            oldx1 = r90_rois[:, 0].copy()
            oldx2 = r90_rois[:, 2].copy()
            oldy1 = r90_rois[:, 1].copy()
            oldy2 = r90_rois[:, 3].copy()
            r90_rois[:, 0] = oldy1
            r90_rois[:, 1] = width - oldx2 - 1
            r90_rois[:, 2] = oldy2
            r90_rois[:, 3] = width - oldx1 - 1
            assert (r90_rois[:, 2] >= r90_rois[:, 0]).all()

            rot_rois = r90_rois

        elif i == 1:
            # ------------------rotate180-------------------
            boxes = rois.copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            oldy1 = boxes[:, 1].copy()
            oldy2 = boxes[:, 3].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 1] = heigt - oldy2 - 1
            boxes[:, 2] = width - oldx1 - 1
            boxes[:, 3] = heigt - oldy1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (boxes[:, 3] >= boxes[:, 1]).all()
            rot_rois = boxes

        elif i == 2:
            # ------------------rotate270-------------------
            boxes = rois.copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            oldy1 = boxes[:, 1].copy()
            oldy2 = boxes[:, 3].copy()
            boxes[:, 0] = heigt - oldy2 - 1
            boxes[:, 1] = oldx1
            boxes[:, 2] = heigt - oldy1 - 1
            boxes[:, 3] = oldx2
            rot_rois = boxes
        else:
            rot_rois = rois.copy()

    return labels.reshape(1, -1), rois, flip_rois, rot_rois


def _get_image_blob(roidb, aug):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images + 3)
    aug_inds = np.random.randint(0, high=2, size=num_images + 2)
    flip_inds = np.random.randint(0, high=2, size=num_images)
    rot_inds =  np.random.randint(0, high=4, size=num_images)
    processed_ims = []
    processed_im_scale = []
    processed_ims_flip = []
    processed_ims_rot = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im_flip = im
            im = im[:, ::-1, :]
        else:
            im_flip = im[:, ::-1, :]

        if aug_inds[0] == 1:
            im0, _, _ = aug(im, None, None)
        else:
            im0 = im

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        img, im_scale = blob_utils.prep_im_for_blob(
            im0, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])

        if flip_inds == 0:
            im1 = im
        else:
            im1 = im_flip
        if aug_inds[1] == 1:
            im1, _, _ = aug(im1, None, None)

        target_size = cfg.TRAIN.SCALES[scale_inds[i + 1]]
        ims, im_scale = blob_utils.prep_im_for_blob(
            im1, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])

        if aug_inds[2] == 1:
            #imt = im_flip
            im_flip, _, _ = aug(im_flip, None, None)
        # else:
        #     imt, _, _ = aug(im_flip, None, None)

        target_size = cfg.TRAIN.SCALES[scale_inds[i + 2]]
        im_flip, im_scale = blob_utils.prep_im_for_blob(
            im_flip, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale[0])

        if rot_inds == 0:
            im_90 = im.transpose(1, 0, 2)
            imt = im_90[::-1, :, :]
        elif rot_inds == 1:
            imt = im[::-1, ::-1, :]
        elif rot_inds == 2:
            im_270 = im.transpose(1, 0, 2)
            imt = im_270[:, ::-1, :]
        else:
            imt = im

        target_size = cfg.TRAIN.SCALES[scale_inds[i + 3]]
        im_rot, im_scale = blob_utils.prep_im_for_blob(
            imt, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale[0])
        processed_ims.append(img[0])
        processed_im_scale.append(ims[0])
        processed_ims_flip.append(im_flip[0])
        processed_ims_rot.append(im_rot[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    blob_scale = blob_utils.im_list_to_blob(processed_im_scale)
    blob_flip = blob_utils.im_list_to_blob(processed_ims_flip)
    blob_rot = blob_utils.im_list_to_blob(processed_ims_rot)

    return blob, blob_scale, blob_flip, blob_rot, im_scales, flip_inds, rot_inds

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois
