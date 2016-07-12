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
                imgs.append(img)
                img_masks.append(img_mask)
                
                #Augment data
                trans_imgs, trans_img_masks = augment_data(img,img_mask,patient_idx,image_idx)
                for trans_img in trans_imgs:
                    imgs.append(trans_img)
                for trans_img_mask in trans_img_masks: 
                    img_masks.append(trans_img_mask)             
                print "patient {}: loaded image {}".format(patient_idx+1,image_idx+1)
                prev_img = img
                prev_mask = img_mask
            except:
                imgs.append(prev_img)
                img_masks.append(prev_mask)
                trans_imgs, trans_img_mask = augment_data(prev_img,prev_mask,patient_idx,image_idx)
                for trans_img in trans_imgs:
                    imgs.append(trans_img)
                for trans_img_mask in trans_img_masks: 
                    img_masks.append(trans_img_mask)  
                print "patient {}: image or mask {} not found, skipping".format(patient_idx+1,image_idx+1)
          
    print "Augmented Training data stats : {} images have been loaded".format(len(imgs))
    imgs = np.array(imgs).reshape(-1, 1, 420, 580)
    img_masks = np.array(img_masks).reshape(-1, 1, 420, 580)
    return imgs,img_masks

def iterate_minibatches(inputs, targets,batchsize, shuffle=False):
    assert len(inputs) == len(targets)
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

def augment_data(img,img_mask,patient_idx,image_idx):
    
    trans_imgs = []
    trans_img_masks = []

    vertical_axis = cv2.flip(img,1)
    v_axis_mask = cv2.flip(img_mask,1)

    horizontal_axis = cv2.flip(img,0)
    h_axis_mask = cv2.flip(img_mask,0)

    both_axis = cv2.flip(vertical_axis,0)
    b_axis_mask = cv2.flip(v_axis_mask,0)

    trans_imgs.append(vertical_axis)
    trans_imgs.append(horizontal_axis)
    trans_imgs.append(both_axis)

    trans_img_masks.append(v_axis_mask)
    trans_img_masks.append(h_axis_mask)
    trans_img_masks.append(b_axis_mask)
    
    return trans_imgs , trans_img_masks
   
"""

    #saving the original and the transformed image + mask
    filename='%s_%s.png'
    filename_mask='%s_%s_mask.png'

    image_array = np.concatenate((img,vertical_axis),axis=0)
    temp = np.concatenate((img_mask,v_axis_mask),axis=0)
    
    toimage(image_array).save(filename % (patient_idx+1,image_idx+1))
    toimage(temp).save(filename_mask % (patient_idx+1,image_idx+1))
"""
    

    
    

def per_patient_train_val_split(dataset , num_train , test_num, num_patients):
    
    

    train_set = np.empty([num_train , 1 , img_rows , img_cols])
    test_set = np.empty([test_num , 1 , img_rows , img_cols])
    per_patient = num_train / num_patients
    test_per_patient = test_num / num_patients

    for each_patient in range(num_patients):
        train_set[each_patient*per_patient:(each_patient+1)*per_patient] = dataset[each_patient*120*4:(each_patient*120*4)+per_patient]
        test_set[each_patient*test_per_patient:(each_patient+1)*test_per_patient] = dataset[((each_patient+1)*120*4) - test_per_patient:(each_patient+1)*120*4]
    
    
    return train_set , test_set 
    


   

        



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
    #val_patients = 5

    test_num = 705 * 4
    num_train = 4935 * 4
    
    X_train , X_val = per_patient_train_val_split(imgs , num_train , test_num , num_patients)
    y_train , y_val = per_patient_train_val_split(img_masks , num_train , test_num, num_patients)


    #X_train, X_val = imgs[:-(val_patients*num_images*4)], imgs[-(val_patients*num_images*4):] 
    #y_train, y_val = img_masks[:-(val_patients*num_images*4)], img_masks[-(val_patients*num_images*4):]
    #X_train = imgs
    #y_train = img_masks


    #normalize data
   
    img_mean = np.mean(X_train)
    img_std = np.mean(X_train)
    X_train=(X_train-img_mean)/img_std
    X_train=X_train.astype('float32')
    X_val=(X_val-img_mean)/img_std
    X_val=X_val.astype('float32')
    y_train=y_train.astype('float32')
    y_val = y_val.astype('float32')

    print("Loading and preprocessing of data took {:.2f} seconds".format(time.time()-start_time))

    print('*'*30)
    print('Creating required network...')
    print('*'*30)
   
    # Create theano variable for inputs and targets 
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
   
    # parameters for the network
    lr = 1e-5
    weight_decay = 0.00005
    
    #build required network
    network = fcn(input_var)

    # get predictions from the network
    prediction = lasagne.layers.get_output(network)
   
    # setup loss and update terms
    params = lasagne.layers.get_all_params(network, trainable=True)
    smooth = 1
    loss = dice_coef_loss(target_var,prediction,smooth)
#    loss = loss + (weight_decay * lasagne.regularization.regularize_network_params(network,
#           lasagne.regularization.l2))
    updates = lasagne.updates.momentum(loss,params,learning_rate=0.1,momentum=0.9)
    
  
    # validation predictions
   
    validation_prediction = lasagne.layers.get_output(network, deterministic=True)


    validation_loss = dice_coef_loss(target_var,validation_prediction,smooth)
   
    #validation nerve presence predictions
 

    #setup training , validation and output functions 
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], validation_loss)
    get_output = theano.function([input_var] , validation_prediction)


    

    num_epochs=30
    train_batchsize=32
    val_batchsize = 32
    train_dices = []
    val_dices = []
    epochs_count = []
    print("Starting training...")
    
    #iterate over epochs
    for epoch in range(num_epochs):
        train_dice = 0
        train_batches = 0
        start_time = time.time()
        
        print "X_train shape: {}".format(X_train.shape)
        for batch in iterate_minibatches(X_train, y_train,train_batchsize, shuffle=True):
            inputs, targets = batch
            train_dice += train_fn(inputs, targets)            
            train_batches += 1
       
        #Run over validation data
        val_dice = 0
        val_batches = 0 

        for btch in iterate_minibatches(X_val, y_val,val_batchsize, shuffle=True):
            inputs, targets = btch
            val_dice += val_fn(inputs, targets)
            
            val_batches += 1

         # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training dice coefficient:\t\t{:.3f}".format(-1*train_dice/(train_batches)))   
        print("  validation set dice coefficent:\t\t{:.3f} ".format(-1*val_dice/(val_batches)))
       
        
        train_dices.append(-1*train_dice/(train_batches))
        val_dices.append(-1*val_dice/(val_batches))
        epochs_count.append(epoch+1)
        npz_name = str(epoch+1) + "_epochs_segnet_flip.npz"                   
        np.savez(npz_name,*lasagne.layers.get_all_param_values(network))
        print("Saved the model after the {}th epoch".format(epoch+1))

       

    #compute output of first test_sample
    #pred = get_output(X_val[:10])


   # pp = PdfPages('./Masks16June.pdf');

    #for out_idx in range(10):
     #   fig, axis1 =plt.subplots()

      #  plt.imshow(pred[out_idx,0,...])
       # pp.savefig(fig)




   # pp.close()

#   Plot Training Curve
    plt.plot(epochs_count, train_dices , 'ro' , ls = '-' , label = 'Training')
    plt.plot(epochs_count, val_dices , 'go' , ls='-', label= 'Validation')
    plt.legend(loc='upper left')
    plt.xlabel("Number of Epochs (Time)")
    plt.ylabel("Dice coefficient (Higher is better)")
    plt.savefig('Training_Curve.png')


if __name__ == "__main__":
    train()

