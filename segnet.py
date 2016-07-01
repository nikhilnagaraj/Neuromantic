import os
import numpy as np
import theano
import theano.tensor as T
import random

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import lasagne

import time
from PIL import Image

from scipy.misc import toimage
import scipy.misc
import skimage.transform
import cv2

#Reproduce results
np.random.seed(123)


img_rows = 128
img_cols = 160


#theano.config.profile = True
#os.environ['CUDA_LAUNCH_BLOCKING']= '1'
#theano.config.allow_gc = False

def dice_coef(y_true, y_pred, smooth):

    """ Calculates the Dice Coefficient.
     
    Parameters
    ----------
    y_true : The actual mask of a image
    y_pred : The predicted mask 
    smooth : parameter to ensure the dice coefficients doesn't fall into either extreme.
    """

    y_true_f = T.flatten(y_true,outdim=2)
    y_pred_f = T.flatten(y_pred,outdim=2)
    intersection = T.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) +smooth)


def dice_coef_loss(y_true, y_pred, smooth):
    return -dice_coef(y_true,y_pred,smooth)

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
    #random.seed(44)
    imgs=[]
    img_masks=[]
    #mask_presence = []
    filename='%s_%s.tif'
    filename_mask='%s_%s_mask.tif'

    for patient_idx in range(num_patients):
        for image_idx in range(num_images):
            try:
                img=plt.imread(filepath+filename % (patient_idx+1,image_idx+1))
                img_mask=plt.imread(filepath+filename_mask % (patient_idx+1,image_idx+1))
                #if np.count_nonzero(img_mask) > 0:
                #    mask_presence.append(1)
                #else:
                #    mask_presence.append(0)
                imgs.append(img)
                img_masks.append(img_mask)
                
                #Randomly augment data
                #if random.choice([True,False]):
                #    trans_img, trans_img_mask = augment_data(img,img_mask)
                #    imgs.append(trans_img)
                #    img_masks.append(trans_img_mask)
                #    if np.count_nonzero(trans_img_mask) > 0:
                #        mask_presence.append(1)
                #    else:
                #        mask_presence.append(0)                
                print "patient {}: loaded image {}".format(patient_idx+1,image_idx+1)
            except:
                print "patient {}: image or mask {} not found, skipping".format(patient_idx+1,image_idx+1)
          
    print "Training data stats : {} images have been loaded".format(len(imgs))
    imgs = np.array(imgs).reshape(-1, 1, 420, 580)
    img_masks = np.array(img_masks).reshape(-1, 1, 420, 580)
    #mask_presence = np.array(mask_presence).reshape(-1,1)
    #mask_presence = mask_presence.astype(np.uint8)
    #print "Training data stats: {} images out of {} have the brachial plexus ".format(np.count_nonzero(mask_presence),len(imgs))
    return imgs,img_masks

def iterate_minibatches(inputs, targets,batchsize, shuffle=False):
    assert len(inputs) == len(targets)
#    assert len(inputs) == len(other_targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def fast_warp(img, tf_params):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf_params.params
    return skimage.transform._warps_cy._warp_fast(img, m)

def augment_data(img,img_mask):
    shear_param = np.random.uniform(0,0.523) #Randomly choose a shearing angle between 0 and 30 degrees
    #Setup the transformation parameters
    tf_params = skimage.transform.AffineTransform(shear=shear_param)
    trans_img = fast_warp(img , tf_params)
    trans_img_mask = fast_warp(img_mask, tf_params)
    return trans_img , trans_img_mask
"""
    #saving the original and the transformed image + mask
    image_array = np.concatenate((img,trans_img),axis=0)
    temp = np.concatenate((img_mask,trans_img_mask),axis=0)
    
    toimage(image_array).save(str(shear_param) +'.png')
    toimage(temp).save(str(shear_param) +'_mask.png')
"""

 
    
    


        



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

    
    
    
   
   

def train():

       

    #Load and preprocess the images
    print('*'*30)
    print("Loading and preprocessing data...")   
    print('*'*30)
    start_time = time.time()
    #adjust basepath accordingly
    data_basepath='/mnt/storage/users/mfreiber/nerveseg/'
   
    filepath=data_basepath+'train/'
    #number of patients' data to be imported
    num_patients=47
    num_images=120

    imgs,img_masks =load_patient_data(filepath, num_patients, num_images)
    imgs = preprocess(imgs)
    img_masks = preprocess(img_masks)

    #Format mask data

    img_masks = img_masks.astype('float32')    
    img_masks /= 255.

    #Split into training and validation data
    val_patients = 5

    X_train, X_val = imgs[:-(val_patients*num_images)], imgs[-(val_patients*num_images):] 
    y_train, y_val = img_masks[:-(val_patients*num_images)], img_masks[-(val_patients*num_images):]
    #m_p_train , m_p_val = m_presence[:-(val_patients*num_images)], m_presence[-(val_patients*num_images):]
    #X_train = imgs
    #y_train = img_masks


    #normalize data
   
    img_mean = np.mean(X_train)
    img_std = np.mean(X_train)
    X_train=(X_train-img_mean)/img_std
    X_train=X_train.astype('float32')
    X_val=(X_val-img_mean)/img_std
    X_val=X_val.astype('float32')

    print("Loading and preprocessing of data took {:.2f} seconds".format(time.time()-start_time))

    print('*'*30)
    print('Creating required network...')
    print('*'*30)
   
    # Create theano variable for inputs and targets 
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    #mask_presence_var = T.icol('mask_pres')

    # parameters for the network
    lr = 1e-5
    weight_decay = 0.00005
    #mask_pres_lr = theano.shared(np.array(1e-6,dtype=theano.config.floatX))

    #build required network
    network = fcn(input_var)

    # get predictions from the network
    prediction = lasagne.layers.get_output(network)
    #mask_presence_prediction = lasagne.layers.get_output(yn_network)
   
    # setup loss and update terms
    params = lasagne.layers.get_all_params(network, trainable=True)
    smooth = 1
    loss = dice_coef_loss(target_var,prediction,smooth)
#    loss = loss + (weight_decay * lasagne.regularization.regularize_network_params(network,
#           lasagne.regularization.l2))
    updates = lasagne.updates.momentum(loss,params,learning_rate=0.1,momentum=0.9)
    
    #setup loss and update for mask (nerve) presence predictor
#    mask_presence_params = lasagne.layers.get_all_params(yn_network, trainable=True)
#    mask_pres_loss = lasagne.objectives.binary_crossentropy(mask_presence_prediction, mask_presence_var).mean()
    #mask_pres_loss = mask_pres_loss + (1e-4 * lasagne.regularization.regularize_network_params(yn_network, lasagne.regularization.l2))
#    mask_pres_updates = lasagne.updates.adam(mask_pres_loss, mask_presence_params,learning_rate=mask_pres_lr)
       

    # validation predictions
   
    validation_prediction = lasagne.layers.get_output(network, deterministic=True)
#    validation_mask_pres_pred = lasagne.layers.get_output(yn_network, deterministic=True)


    validation_loss = dice_coef_loss(target_var,validation_prediction,smooth)
   
    #validation nerve presence predictions
 
  
#    validation_mask_pres_loss = lasagne.objectives.binary_crossentropy(validation_mask_pres_pred, mask_presence_var).mean()

    #setup training , validation and output functions for both presence and location
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], validation_loss)
    get_output = theano.function([input_var] , validation_prediction)

#    train_pres_fn = theano.function([input_var, mask_presence_var] , mask_pres_loss, updates = mask_pres_updates)
#    val_pres_fn = theano.function([input_var, mask_presence_var], mask_pres_loss)
    

    num_epochs=25
    train_batchsize=32
    val_batchsize = 32
    #best_mask_val_score = 99.99
    print("Starting training...")

    #iterate over epochs
    for epoch in range(num_epochs):
        train_dice = 0
        train_batches = 0
        #train_mask_pres = 0
        start_time = time.time()
        
        print "X_train shape: {}".format(X_train.shape)
        for batch in iterate_minibatches(X_train, y_train,train_batchsize, shuffle=True):
            inputs, targets = batch
            train_dice += train_fn(inputs, targets)
            #train_mask_pres += train_pres_fn(inputs, other_targets)
            train_batches += 1
       
        #Run over validation data
        val_dice = 0
        val_batches = 0 
        #val_mask_pres = 0

        for btch in iterate_minibatches(X_val, y_val,val_batchsize, shuffle=True):
            inputs, targets = btch
            val_dice += val_fn(inputs, targets)
            #val_mask_pres += val_pres_fn(inputs, other_targets)
            val_batches += 1

         # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training dice coefficient:\t\t{:.3f}".format(-1*train_dice/(train_batches)))   
        print("  validation set dice coefficent:\t\t{:.3f} ".format(-1*val_dice/(val_batches)))
        #print("  Training mask prediction loss:\t\t{:.3f} ".format(train_mask_pres/(train_batches)))
        #print("  Validation mask prediction loss:\t\t{:.3f} ".format(val_mask_pres/(val_batches)))
        
       
        #mask_pres_lr = theano.shared(1e-5/(2**(epoch+1)))

        np.savez('forum_model.npz',*lasagne.layers.get_all_param_values(network))
        print("Saved the model after the {}th epoch".format(epoch+1))

        #if(best_mask_val_score > val_mask_pres/val_batches): 
        #    np.savez('forum_model_yn.npz',*lasagne.layers.get_all_param_values(yn_network))
        #    print("Saved the model after the {}th epoch".format(epoch+1))
        #    best_mask_val_score = val_mask_pres/val_batches

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

