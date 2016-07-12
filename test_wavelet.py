import numpy as np
import theano
import theano.tensor as T
import pywt


import scipy
from scipy.misc import toimage
import scipy.misc


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

img_rows = 64
img_cols = 80
custom_rectify = lasagne.nonlinearities.LeakyRectify(0.6)


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

def preprocess_images(imgs):

    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            imgs_p[i,j] = scipy.misc.imresize(imgs[i,j], (img_rows, img_cols),interp ='cubic')       
    return imgs_p 


def postprocess(img,presence_pred_prob):
   
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
   
    component_strings=['a','h','v','d']
   
    for image_idx in range(num_images):
        try:
            channels=[]
            img=plt.imread(filepath+filename % (image_idx+1))
            #imgs.append(img) # we only need the first layer 
            channels.append(img)
            wp=pywt.WaveletPacket2D(data=img,wavelet='db1',mode='sym')
                

            for stage1_idx in range(len(component_strings)):

                for stage2_idx in range(len(component_strings)):

                    for stage3_idx in range(len(component_strings)):

                        wavelet_string=component_strings[stage1_idx]+component_strings[stage2_idx]+component_strings[stage3_idx]
                        img_trans = wp[wavelet_string].data
                           # print "img_trans.shape: {}".format(img_trans.shape)
                        img_trans=scipy.misc.imresize(img_trans, (img_rows, img_cols),interp ='cubic')
                            #img_trans=cv2.resize(img_trans, (580, 420))     
                        channels.append(img_trans)
                            #print "wavelet shape: {}".format(wp[wavelet_string].data.shape)
                            #fig=plt.figure()
                            #plt.imshow(wp[wavelet_string].data)


            imgs.append(channels)
            print " loaded image {}".format(image_idx+1)
        except:
            print " image or mask {} not found, skipping".format(image_idx+1)

    imgs = np.array(imgs)
    return imgs

num_wave_comps=64

def fcn(input_var=None):

    input = lasagne.layers.InputLayer(shape=(None,num_wave_comps+1,img_rows,img_cols), input_var = input_var)
     


    conv1 = lasagne.layers.Conv2DLayer(input,num_filters = 32, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv1 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv1),num_filters = 32, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    pool1 = lasagne.layers.MaxPool2DLayer(conv1,pool_size=(2,2))

    conv2 = lasagne.layers.Conv2DLayer(pool1,num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv2 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv2),num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv2 = lasagne.layers.Conv2DLayer(conv2,num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2,pool_size=(2,2))

    conv3 = lasagne.layers.Conv2DLayer(pool2,num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv3 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv3),num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv3 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv3),num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    pool3 = lasagne.layers.MaxPool2DLayer(conv3,pool_size=(2,2))

    conv4 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(pool3,p=0.5),num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv4 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv4),num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv4 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv4),num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    pool4 = lasagne.layers.MaxPool2DLayer(conv4,pool_size=(2,2))

    conv5 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(pool4,p=0.5),num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv5 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv5),num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv5 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv5),num_filters = 512, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)

    yn    = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(conv5),num_units = 1,W = lasagne.init.GlorotUniform(),nonlinearity=lasagne.nonlinearities.sigmoid)
   
    up6 = lasagne.layers.ConcatLayer((lasagne.layers.Upscale2DLayer(lasagne.layers.DropoutLayer(conv5,p=0.5),(2,2)),conv4),axis=1)
    conv6 = lasagne.layers.Conv2DLayer(up6,num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv6 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv6),num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv6 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv6),num_filters = 256, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)

    up7 = lasagne.layers.ConcatLayer((lasagne.layers.Upscale2DLayer(lasagne.layers.DropoutLayer(conv6,p=0.5),(2,2)),conv3),axis=1)
    conv7 = lasagne.layers.Conv2DLayer(up7,num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv7 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv7),num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv7 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv7),num_filters = 128, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)

    up8 = lasagne.layers.ConcatLayer((lasagne.layers.Upscale2DLayer(lasagne.layers.DropoutLayer(conv7,p=0.5),(2,2)),conv2),axis=1)
    conv8 = lasagne.layers.Conv2DLayer(up8,num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv8 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv8),num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv8 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv8),num_filters = 64, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)

    up9 = lasagne.layers.ConcatLayer((lasagne.layers.Upscale2DLayer(conv8,(2,2)),conv1),axis=1)
    conv9 = lasagne.layers.Conv2DLayer(up9,num_filters = 32, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv9 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv9),num_filters = 32, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    conv9 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv9),num_filters = 32, filter_size=(3,3),stride=1,
              pad='same',W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1),
              nonlinearity = lasagne.nonlinearities.very_leaky_rectify)
    

    conv10 = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(conv9),num_filters=1, filter_size=(1,1),
              nonlinearity = lasagne.nonlinearities.sigmoid,pad='valid',
              W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.1))

    return conv10 , yn



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
    network , yn_network = fcn(input_var)
    
    #Recover previously trained parameters      
    with np.load('forum_model_wavelets.npz') as f:
       param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    with np.load('forum_model_wavelets_yn.npz') as g:
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
    dframe.to_csv("leaky.csv", index=False)

    
if __name__== "__main__":


    data_basepath = '/mnt/storage/users/mfreiber/nerveseg/'
    filepath = data_basepath + 'test/'


    num_images=5508
    batch_size=51
    imgs = load_test_images(filepath, num_images)
    imgs = preprocess_images(imgs)

    for i in range(num_images/batch_size):
        start = batch_size*i
        end = (batch_size*(i+1))
        print("Testing batch {} ...".format(i))
        test(imgs[start:end])

    write_submission_file(num_images)

