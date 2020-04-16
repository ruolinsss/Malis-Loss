MALIS

Structured loss function for supervised learning of segmentation and clustering

Python and MATLAB wrapper for C++ functions for computing the MALIS loss

The MALIS loss is described here:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). Maximin learning of image segmentation. Advances in Neural Information Processing Systems (NIPS) 2009.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation


Preference:
https://github.com/TuragaLab/malis
https://github.com/ELEKTRONN/malis


Installation:

./make.sh            (Building c++ extension only: run inside directory)
pip install .        (Installation as python package: run inside directory)


Installation example in anaconda:
conda create -n malis_test python=3.7
conda install cython
conda install numpy
conda install gxx_linux-64
conda install -c anaconda boost
./make.sh
pip install .


Usage:

An example of using this package is shown in example.ipynb in test folder.

Detailly:
from malis.malis_utils import mknhood3d,seg_to_affgraph,affgraph_to_seg  ## most important 3 functions
from malis.malis_tf import malis_loss

mknhood3d(): Makes neighbourhood structures
seg_to_affgraph(seg_gt,nhood): Construct an affinity graph from a segmentation
affgraph_to_seg(affinity,nhood,size_thresh): Obtain a segentation graph from an affinity graph
loss = malis_loss(aff_pred,aff_gt,seg_gt,nhood): Obtain malis loss


