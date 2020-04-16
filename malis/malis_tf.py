import numpy as np
import tensorflow as tf
from malis_utils import malis_weights_op

def malis_loss(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask=None,
        gt_seg_unlabelled=None,
        name=None):
    
    '''Returns a tensorflow op to compute the constrained MALIS loss, using the
    squared distance to the target values for each edge as base loss.
    In the simplest case, you need to provide predicted affinities (``affs``),
    ground-truth affinities (``gt_affs``), a ground-truth segmentation
    (``gt_seg``), and the neighborhood that corresponds to the affinities.
    This loss also supports masks indicating unknown ground-truth. We
    distinguish two types of unknowns:
        1. Out of ground-truth. This is the case at the boundary of your
           labelled area. It is unknown whether objects continue or stop at the
           transition of the labelled area. This mask is given on edges as
           argument ``gt_aff_mask``.
        2. Unlabelled objects. It is known that there exists a boundary between
           the labelled area and unlabelled objects. Withing the unlabelled
           objects area, it is unknown where boundaries are. This mask is also
           given on edges as argument ``gt_aff_mask``, and with an additional
           argument ``gt_seg_unlabelled`` to indicate where unlabelled objects
           are in the ground-truth segmentation.
    Both types of unknowns require masking edges to exclude them from the loss:
    For "out of ground-truth", these are all edges that have at least one node
    inside the "out of ground-truth" area. For "unlabelled objects", these are
    all edges that have both nodes inside the "unlabelled objects" area.
    Args:
        affs (Tensor): The predicted affinities.
        gt_affs (Tensor): The ground-truth affinities.
        gt_seg (Tensor): The corresponding segmentation to the ground-truth
            affinities. Label 0 denotes background.
        neighborhood (Tensor): A list of spatial offsets, defining the
            neighborhood for each voxel.
        gt_aff_mask (Tensor): A binary mask indicating where ground-truth
            affinities are known (known = 1, unknown = 0). This is to be used
            for sparsely labelled ground-truth and at the borders of labelled
            areas. Edges with unknown affinities will not be constrained in the
            two malis passes, and will not contribute to the loss.
        gt_seg_unlabelled (Tensor): A binary mask indicating where the
            ground-truth contains unlabelled objects (labelled = 1, unlabelled
            = 0). This is to be used for ground-truth where only some objects
            have been labelled. Note that this mask is a complement to
            ``gt_aff_mask``: It is assumed that no objects cross from labelled
            to unlabelled, i.e., the boundary is a real object boundary.
            Ground-truth affinities within the unlabelled areas should be
            masked out in ``gt_aff_mask``. Ground-truth affinities between
            labelled and unlabelled areas should be zero in ``gt_affs``.
        name (string, optional): A name to use for the operators created.
    Returns:
        A tensor with one element, the MALIS loss.
    '''
    
    if gt_aff_mask is None:
        gt_aff_mask = tf.zeros((0,))
    if gt_seg_unlabelled is None:
        gt_seg_unlabelled = tf.zeros((0,))

    weights_neg,weights_pos = tf.py_function(malis_weights_op,
                             [affs, gt_affs, gt_seg, neighborhood,gt_aff_mask,gt_seg_unlabelled],
                             [tf.int32,tf.int32],
                              name = name)
    
    pos_t = tf.cast(weights_pos,dtype=tf.float32)
    pos_t = tf.math.divide_no_nan(pos_t,tf.reduce_sum(pos_t))

    neg_t = tf.cast(weights_neg,dtype=tf.float32)
    neg_t = tf.math.divide_no_nan(neg_t,tf.reduce_sum(neg_t))
    weights = pos_t + neg_t
    
    gt_affs = tf.cast(gt_affs,tf.float32)
    edge_loss = tf.square(tf.subtract(gt_affs, affs))

    return tf.reduce_sum(tf.multiply(weights, edge_loss))
