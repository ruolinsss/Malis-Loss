
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from malis import malis_utils


print("Malis TESTING/DEMO 2:")

nhood = np.array([[0., 1., 0.], [0., 0., 1.]], dtype=np.int32)

test_id2 = np.array([[[1, 1, 2, 2, 0, 0, 3, 3], [1, 1, 2, 2, 0, 0, 3, 3],
                      [1, 1, 2, 2, 0, 0, 3, 3], [1, 1, 2, 2, 0, 0, 3, 3]]],
                    dtype=np.int32)

aff_gt = malis_utils.seg_to_affgraph(test_id2, nhood)
seg_gt = malis_utils.affgraph_to_seg(aff_gt, nhood)[0].astype(np.int16)
aff_pred = np.array([[[[1., 1., 1., 1., 0., 0., 1., 1.],
                       [1., 1., 1., 1., 0., 0., 1., 1.],
                       [0.9, 0.8, 1., 1., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 0., 0., 1., 1.]]],

                     [[[1., 0., 1., 0.3, 0.2, 0.3, 1., 0.],
                       [0.7, 0., 1., 0., 0., 0., 1., 0.],
                       [1., 0.2, 1., 0., 0., 0., 1., 0.],
                       [1., 0., 1., 0., 0., 0., 1., 0.]]]]).astype(np.float32)

pos_t, neg_t = tf.py_function(
            malis_utils.malis_weights,
            [aff_pred, aff_gt, seg_gt, nhood],
            [tf.int32,tf.int32])
pos_t = tf.cast(pos_t,tf.float32)
neg_t = tf.cast(neg_t,tf.float32)
loss = tf.reduce_sum(pos_t * aff_pred)



################ Checking ################
pos = [[[[ 1, 3, 4, 0, 0, 0, 12, 0],
   [ 8, 0, 0, 16, 0, 0, 0, 8],
   [12, 0, 0, 4, 0, 0, 0, 4],
   [ 0, 0, 0, 0, 0, 0, 0, 0]]],
 [[[ 2, 0, 1, 0, 0, 0, 1, 0],
   [ 0, 0, 1, 0, 0, 0, 1, 0],
   [ 1, 0, 1, 0, 0, 0, 1, 0],
   [ 1, 0, 1, 0, 0, 0, 1, 0]]]]
neg = [[[[  0, 0, 0, 0, 0, 0, 0, 0],
   [  0, 0,  0,   0,  0, 0,  0,  0],
   [  0, 0,  0,   0,  0,  0,  0,  0],
   [  0, 0,  0,   0,  0,  0,  0,  0]]],
 [[[  0, 0,  0,   0, 64,  0,  0,  0],
   [  0, 0,  0,   0,  0,  0,  0,  0],
   [  0, 128,  0, 0,  0,  0,  0,  0],
   [  0, 0,  0,   0,  0,  0,  0,  0]]]]

assert abs(loss - 82.8) < 0.1, "malis loss is calculated incorrectly"
assert (pos_t.numpy() == pos).all, "pos counts are incorrectly"
assert (pos_t.numpy() == neg).all, "neg counts are incorrectly"

print('Successfully!')

