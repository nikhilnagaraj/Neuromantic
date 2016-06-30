import os
import numpy as np
import theano
import theano.tensor as T

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear, sigmoid

import time
from PIL import Image

import scipy.misc
import cv2

#Reproduce results
np.random.seed(123)

img_rows = 128
img_cols = 160


#theano.config.profile = True
#os.environ['CUDA_LAUNCH_BLOCKING']= '1'
theano.config.allow_gc = False


def preprocess(imgs):

    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i,0] = scipy.misc.imresize(imgs[i,0], (img_rows, img_cols),interp ='cubic')
    return imgs_p 

def load_patient_data(filepath, num_patients, num_images):

    """ Load all the images from a set of patients into a 
    numpy array for further processing

    Parameters
    ----------
    filepath : string
    Path to the data directory
    num_patients : int
    Number of patients for which data shall be loaded
    """

    imgs=[]
    mask_presence = []
    filename='%s_%s.tif'
    filename_mask='%s_%s_mask.tif'

    for patient_idx in range(num_patients):
        for image_idx in range(num_images):
            try:
                img=plt.imread(filepath+filename % (patient_idx+1,image_idx+1))
                img_mask=plt.imread(filepath+filename_mask % (patient_idx+1,image_idx+1))
                if np.count_nonzero(img_mask) > 0:
                    mask_presence.append(1)
                else:
                    mask_presence.append(0)
                imgs.append(img) # we only need the first layer  
                print "patient {}: loaded image {}".format(patient_idx+1,image_idx+1)
            except:
                print "patient {}: image or mask {} not found, skipping".format(patient_idx+1,image_idx+1)

    imgs = np.array(imgs).reshape(-1, 1, 420, 580)
    mask_presence = np.array(mask_presence).reshape(-1,1).astype(np.uint8)
    print "Training data stats: {:.2f}% images have the brachial plexus ".format((np.count_nonzero(mask_presence)/mask_presence.shape[0]) * 100)
    return imgs,mask_presence

def iterate_minibatches(inputs, other_targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    assert len(inputs) == len(other_targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], other_targets[excerpt]


def build_inception_module(name,input_layer, num_filters):

#   name = Name of the inception module
#   input_layer = THe layer that is to be taken as the input
#   num_filters = An array consisting of num of filters for each module
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], num_filters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, num_filters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, num_filters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], num_filters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, num_filters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], num_filters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def network(input_var=None):

    net = {}
    net['input'] = InputLayer((None, 1, img_rows, img_cols),input_var=input_var)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(
      net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(
      net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(
      net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['dropout/40%'] = DropoutLayer(net['pool5/7x7_s1'],p=0.4)
    net['loss3/classifier'] = DenseLayer(net['dropout/40%'],
                                         num_units=1000,
                                         nonlinearity=linear)
    output = DenseLayer(net['loss3/classifier'],num_units=1,nonlinearity = sigmoid)

    return output

def train():

       

    #Load and preprocess the images
    print('*'*30)
    print("Loading and preprocessing data...")   
    print('*'*30)

    #adjust basepath accordingly
    data_basepath='/mnt/storage/users/mfreiber/nerveseg/'
   
    filepath=data_basepath+'train/'
    #number of patients' data to be imported
    num_patients=2
    num_images=120

    imgs, m_presence=load_patient_data(filepath, num_patients, num_images)
    imgs = preprocess(imgs)
   
    #Format mask data

   
    #Split into training and validation data
    val_patients = 5

    X_train, X_val = imgs[:-(val_patients*num_images)], imgs[-(val_patients*num_images):] 
    m_p_train , m_p_val = m_presence[:-(val_patients*num_images)], m_presence[-(val_patients*num_images):]
    #X_train = imgs
    #y_train = img_masks


    #normalize data
   
    img_mean = np.mean(X_train)
    img_std = np.mean(X_train)
    X_train=(X_train-img_mean)/img_std
    X_train=X_train.astype('float32')
    X_val=(X_val-img_mean)/img_std
    X_val=X_val.astype('float32')


    print('*'*30)
    print('Creating required network...')
    print('*'*30)
   
    # Create theano variable for inputs and targets 
    input_var = T.tensor4('inputs')
    mask_presence_var = T.icol('mask_pres')

    # parameters for the network
    lr = 1e-5

    #build required network
    yn_network = network(input_var)

    # get predictions from the network
    mask_presence_prediction = lasagne.layers.get_output(yn_network)
   
    #setup loss and update for mask (nerve) presence predictor
    mask_presence_params = lasagne.layers.get_all_params(yn_network, trainable=True)
    mask_pres_loss = lasagne.objectives.binary_crossentropy(mask_presence_prediction,mask_presence_var).mean()
    mask_pres_updates = lasagne.updates.adam(mask_pres_loss, mask_presence_params,learning_rate=lr)
       

    # validation predictions
   
    validation_mask_pres_pred = lasagne.layers.get_output(yn_network, deterministic=True)

    #validation nerve presence predictions
 
  
    validation_mask_pres_loss = lasagne.objectives.binary_crossentropy(validation_mask_pres_pred, mask_presence_var)

    #setup training , validation and output functions for both presence and location
    
    train_pres_fn = theano.function([input_var, mask_presence_var] , mask_pres_loss, updates = mask_pres_updates)
    val_pres_fn = theano.function([input_var, mask_presence_var], mask_pres_loss)
    

    num_epochs=20
    train_batchsize=32
    val_batchsize = 32

    print("Starting training...")

    #iterate over epochs
    for epoch in range(num_epochs):
        train_batches = 0
        train_mask_pres = 0
        start_time = time.time()
        
        print "X_train shape: {}".format(X_train.shape)
        for batch in iterate_minibatches(X_train, m_p_train, train_batchsize, shuffle=True):
            inputs, other_targets = batch
            train_mask_pres += train_pres_fn(inputs, other_targets)
            train_batches += 1
       
        #Run over validation data
        
        val_batches = 0 
        val_mask_pres = 0

        for btch in iterate_minibatches(X_val, m_p_val, val_batchsize, shuffle=True):
            inputs, other_targets = btch
            
            val_mask_pres += val_pres_fn(inputs, other_targets)
            val_batches += 1

         # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  Training mask prediction loss:\t\t{:.3f} ".format(train_mask_pres/(train_batches)))
        print("  Validation mask prediction loss:\t\t{:.3f} ".format(val_mask_pres/(val_batches)))

        np.savez('forum_model_yn.npz',*lasagne.layers.get_all_param_values(yn_network))
        print("Saved the model after the {}th epoch".format(epoch+1))

    #compute output of first test_sample
    #pred = get_output(X_val[:10])


   # pp = PdfPages('./Masks16June.pdf');

    #for out_idx in range(10):
     #   fig, axis1 =plt.subplots()

      #  plt.imshow(pred[out_idx,0,...])
       # pp.savefig(fig)




   # pp.close()


if __name__ == "__main__":
    train()

