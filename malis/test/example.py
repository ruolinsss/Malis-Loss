
# coding: utf-8

# In[1]:


### use keras with backend tensorflow as a test example
### tensorflow version 2.1.0

import keras
import keras.backend as K
from keras.layers import Input,Conv3D,MaxPooling3D,UpSampling3D,Lambda,Activation,concatenate
from keras.models import Model,Sequential
from keras.callbacks import CSVLogger
import tensorflow as tf
import numpy as np
import h5py
from malis.malis_utils import mknhood3d,seg_to_affgraph,affgraph_to_seg
from malis.malis_tf import malis_loss


# In[2]:


##### Loading test data (can be downloaded on https://cremi.org/data/)
f = h5py.File('./test_data/sample_A_20160501.hdf','r')
raw_data = f['volumes']['raw']
seg_gt = f['volumes']['labels']['neuron_ids']

##### Data preprocessing, affinity ground truth should be prepared
nhood = mknhood3d(1)
e = nhood.shape[0]
aff_gt = seg_to_affgraph(seg_gt, nhood)   #The seg_gt needs to be reshaped as (z,y,x) and the output aff_gt has shape of (edge,z,y,x) 

# adding dimensions to have 5d input data (batch,channel,x,y,z)
data_ch = np.expand_dims(np.expand_dims(data,axis=0),axis=1)   #(batch,channel=1,z,y,x)
aff_gt_label = np.expand_dims(aff_gt_label,axis=0)             #(batch,channel=edge,z,y,x)
seg_gt = np.expand_dims(np.expand_dims(seg_gt,axis=0),axis=1)  #(batch,channel=1, z,y,x)


# In[3]:


###### Loss example for keras (tensorflow backend)
def MALIS_loss(seg_gt):
    
    def loss(y_true,y_pred):
        
        z = K.int_shape(y_pred)[2]
        y = K.int_shape(y_pred)[3]
        x = K.int_shape(y_pred)[4]

        new_y_true = K.reshape(y_true,(e,-1,y,x))
        new_y_pred = K.reshape(y_pred,(e,-1,y,x))
        new_seg = K.reshape(seg_gt,(-1,y,x))
        
        loss = malis_loss(new_y_pred, new_y_true, new_seg, nhood)
        
        return loss
    return loss


# In[4]:


##### A very simple network example
inputShape = (data_ch.shape[1], data_ch.shape[2], data_ch.shape[3], data_ch.shape[4])

inputs = Input(shape=(inputShape))
input_seg = Input(shape=(inputShape))
conv_block_1 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
conv_block_1 = Activation('relu')(conv_block_1)
pool_block_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_1)

conv_block_2 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(pool_block_1)
conv_block_2 = Activation('relu')(conv_block_2)
pool_block_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block_2)


up_block_1 = UpSampling3D((2, 2, 2))(pool_block_2)
up_block_1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_1)
up_block_2 = UpSampling3D((2, 2, 2))(up_block_1)
up_block_2 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same')(up_block_2)
conv_block_10 = Conv3D(e, (1, 1, 1), strides=(1, 1, 1), padding='same')(up_block_2) #### output of the network is affinity graph
outputs = Activation('sigmoid')(conv_block_10)

##### raw data, segmentation label and affinity label should be input to the network
model = Model(inputs=[inputs,input_seg], outputs=outputs)
model.compile(optimizer='adadelta', loss=MALIS_loss(input_seg))
model.summary()
model.fit([data_ch,seg_gt],aff_gt_label,epochs=3,verbose=1,batch_size=1,validation_split = 0.1)


# In[ ]:


###### Data postprocessing: get segmentation image from affinity graph
seg_pred = affgraph_to_seg(aff_pred,nhood,size_thresh=1)

