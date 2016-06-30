import numpy as np
import theano
import theano.tensor as T

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lasagne

import pandas as pd

import time
from PIL import Image
import cv2
#import scipy.misc
#from skimage.filters import threshold_otsu
#from skimage import transform as tf

rle_mask = []

img_rows = 128
img_cols = 160


def dice_coef(y_true, y_pred, smooth):

    """ Calculates the Dice Coefficient.

    Parameters
    ----------
    y_true : The actual mask of a image
    y_pred : The predicted mask 
    smooth : parameter to ensure the dice coefficients doesn't fall into either extreme.
    """

    y_true_f = T.flatten(y_true)
    y_pred_f = T.flatten(y_pred)
    intersection = T.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) +smooth)

def dice_coef_loss(y_true, y_pred, smooth):
    return -dice_coef(y_true,y_pred,smooth)

def preprocess(imgs):

    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i,0] = cv2.resize(imgs[i,0], (img_cols, img_rows),interpolation = cv2.INTER_CUBIC)
    return imgs_p

def postprocess(img, presence_pred_prob):
   
    img = img.astype('float32')
    img *= presence_pred_prob
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (580, 420))
    return img

from itertools import chain


def Rlenc(label):

    """ Takes a mask and cencodes it using Run length encoding

    """

    x = label.transpose().flatten()
    y = np.where(x>0)[0]
    if len(y)<10:
        return ''
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    length = end - start
    res = [[s+1,l+1] for s,l in zip(list(start),list(length))]
    res = list(chain.from_iterable(res))
    return res


def load_test_images(filepath, num_images):

    """ Load all the images from a set of patients into a 
    numpy array for further processing

    Parameters
    ----------
    filepath : string
    Path to the data directory
    num_images = total number of test images
   
    """

    imgs=[]
    filename='%s.tif'
   

   
    for image_idx in range(num_images):
        try:
            img=plt.imread(filepath+filename % (image_idx+1))
            imgs.append(img) # we only need the first layer 
            print " loaded image {}".format(image_idx+1)
        except:
            print " image or mask {} not found, skipping".format(image_idx+1)

    imgs = np.array(imgs).reshape(-1, 1, 420, 580)
    return imgs

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


def pres_network(input_var=None):

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


def fcn(input_var=None):

    input = lasagne.layers.InputLayer(shape=(None,1,img_rows,img_cols), input_var = input_var)
     
    conv1 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(input,num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv1 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv1,num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    pool1 = lasagne.layers.MaxPool2DLayer(conv1,pool_size=(2,2))

    conv2 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(pool1,num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv2 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv2,num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    pool2 = lasagne.layers.MaxPool2DLayer(conv2,pool_size=(2,2))

    conv3 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(pool2,num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv3 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv3,num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv3 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv3,num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    pool3 = lasagne.layers.MaxPool2DLayer(conv3,pool_size=(2,2))

    conv4 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(pool3,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv4 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv4,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv4 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv4,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    pool4 = lasagne.layers.MaxPool2DLayer(conv4,pool_size=(2,2))

    conv5 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(pool4,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv5 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv5,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv5 = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(conv5,num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    pool5 = lasagne.layers.MaxPool2DLayer(conv5,pool_size=(2,2))

    unpool6 = lasagne.layers.Upscale2DLayer(pool5, (2,2))
    conv6 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(unpool6,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv6 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv6,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv6 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv6,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))

    unpool7 = lasagne.layers.Upscale2DLayer(conv6, (2,2))
    conv7 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(unpool7,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv7 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv7,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv7 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv7,num_filters = 512, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))

    unpool8 = lasagne.layers.Upscale2DLayer(conv7, (2,2))
    conv8 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(unpool8,num_filters = 256, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv8 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv8,num_filters = 256, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv8 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv8,num_filters = 256, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))

    unpool9 = lasagne.layers.Upscale2DLayer(conv8, (2,2))
    conv9 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(unpool9,num_filters = 128, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv9 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv9,num_filters = 128, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))

    unpool10 = lasagne.layers.Upscale2DLayer(conv9, (2,2))
    conv10 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(unpool10,num_filters = 64, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))
    conv10 = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(conv10,num_filters = 64, filter_size=(3,3),stride=1,
              crop='same',W=lasagne.init.HeNormal(gain='relu'),b=None,
              nonlinearity = lasagne.nonlinearities.rectify))

    conv11 = lasagne.layers.Conv2DLayer(conv10,num_filters=1, filter_size=(1,1),
              nonlinearity = lasagne.nonlinearities.sigmoid,pad='same',
              W=lasagne.init.HeNormal(),b=None)

    return conv11

    
   
   

def test(imgs):

    global rle_mask

    X_test = imgs
    #normalize data
    img_mean = np.mean(X_test)
    img_std = np.std(X_test)
    X_test = (X_test-img_mean)/img_std
    X_test = X_test.astype('float32')

    #Define input variable
    input_var= T.tensor4('inputs')

    #Build the network
    network = fcn(input_var)
    yn_network = pres_network(input_var)
    
    #Recover previously trained parameters      
    with np.load('forum_model.npz') as f:
       param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    with np.load('forum_model_yn.npz') as g:
       yn_param_values = [g['arr_%d' % i] for i in range(len(g.files))]

#    print param_values
    lasagne.layers.set_all_param_values(network, param_values)
    lasagne.layers.set_all_param_values(yn_network, yn_param_values)

    #Setup prediction variable
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_nerve_presence_prediction = lasagne.layers.get_output(yn_network, deterministic=True)

    #Setup test function
    get_output = theano.function([input_var],test_prediction)
    get_m_pres_output = theano.function([input_var],test_nerve_presence_prediction)


    #Compute output of the test  examples
    pred  = get_output(X_test)
    presence_prediction = get_m_pres_output(X_test)

    #Compute RLE for the predicted masks and add it to  the list of rle masks
    i = 0
    resizedimages = np.empty((pred.shape[0],420,580))
    for img in pred:
        resizedimages[i] = postprocess(img[0] , presence_prediction[i])
    #    resizedimages[i] = img[0] 
        i = i+1

    for predicted_mask in resizedimages:
        rle_mask.append(" ".join(str(r) for r in Rlenc(predicted_mask)))

     #PDf of confidence maps of test images
#    pp = PdfPages('./test_probabilities.pdf');

#    for out_idx in range(len(resizedimages)):
#        fig, axis1 =plt.subplots()

#        plt.imshow(resizedimages[out_idx,0,...])
#        pp.savefig(fig)

#    pp.close()


def write_submission_file(num_images):

    """Uses the run length encoded data to create a submission file according to the format specified 
       on the competition website

    Parameters:    
    num_images = Number of images in the test set

    """
    #Change this later. not good.
    global rle_mask


    indx = np.arange(num_images) + 1
    print("Writing Submission file ...")
    dframe = pd.DataFrame({"img": indx, "pixels":rle_mask})
    dframe.to_csv("segnet_pres.csv", index=False)

    
if __name__== "__main__":


    data_basepath = '/mnt/storage/users/mfreiber/nerveseg/'
    filepath = data_basepath + 'test/'


    num_images=5508
    batch_size=51
    imgs = load_test_images(filepath, num_images)
    imgs = preprocess(imgs)

    for i in range(num_images/batch_size):
        start = batch_size*i
        end = (batch_size*(i+1))
        print("Testing batch {} ...".format(i))
        test(imgs[start:end])

    write_submission_file(num_images)

