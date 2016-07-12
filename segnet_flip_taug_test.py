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

def postprocess(img):
   
    img = img.astype('float32')
    #img *= presence_pred_prob
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
    
    #Recover previously trained parameters      
    with np.load('20_epochs_segnet_flip.npz') as f:
       param_values = [f['arr_%d' % i] for i in range(len(f.files))]

#    with np.load('forum_model_yn.npz') as g:
#       yn_param_values = [g['arr_%d' % i] for i in range(len(g.files))]

#    print param_values
    lasagne.layers.set_all_param_values(network, param_values)
    #lasagne.layers.set_all_param_values(yn_network, yn_param_values)

    #Setup prediction variable
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#    test_nerve_presence_prediction = lasagne.layers.get_output(yn_network, deterministic=True)

    #Setup test function
    get_output = theano.function([input_var],test_prediction)
    #get_m_pres_output = theano.function([input_var],test_nerve_presence_prediction)


    #Compute output of the test  examples
    pred  = get_output(X_test)
#    presence_prediction = get_m_pres_output(X_test)

    return pred
   
     #PDf of confidence maps of test images
#    pp = PdfPages('./test_probabilities.pdf');

#    for out_idx in range(len(resizedimages)):
#        fig, axis1 =plt.subplots()

#        plt.imshow(resizedimages[out_idx,0,...])
#        pp.savefig(fig)

#    pp.close()

def augment_data(img):
    
    
    vertical_axis = cv2.flip(img,1)
    print("   Flipped about vertical axis..")
    
    horizontal_axis = cv2.flip(img,0)
    print("   Flipped about Horizontal axis..")
   
    both_axis = cv2.flip(vertical_axis,0)
    print("   Flipped about both axes..")

    return vertical_axis , horizontal_axis , both_axis   

def write_submission_file(num_images):

    """Uses the run length encoded data to create a submission file according to the format specified 
       on the competition website

    Parameters:    
    num_images = Number of images in the test set

    """
    
    global rle_mask


    indx = np.arange(num_images) + 1
    print("Writing Submission file ...")
    dframe = pd.DataFrame({"img": indx, "pixels":rle_mask})
    dframe.to_csv("20_epochs_flip_taug.csv", index=False)

def unaugment(img, index):

    """Reverses the augemntation applied to the image previously.
       img = Image to unaugment
       index = The type of augmentation
           0 = return image
           1 = FLip about vertical axis
           2 = flip about horizontal axis 
           3 = Flip about horizontal and then vertical axis
    """

    if index == 1 :
        vertical_axis = cv2.flip(img,1)
        print("   Unflipped about vertical axis..")
        return vertical_axis
  
    elif index == 2 :
        horizontal_axis = cv2.flip(img,0)
        print("   Unflipped about horizontal axis..")
        return horizontal_axis

    elif index == 3:
        h_axis = cv2.flip(img,0)
        v_axis = cv2.flip(h_axis,1)
        print("   Unflipped about both axes..")
        return v_axis

    else:
        print("   Kept as such")
        return img            
        
    


def main():
  
    global rle_mask

    data_basepath = '/mnt/storage/users/mfreiber/nerveseg/'
    filepath = data_basepath + 'test/'

#Very very bad implementation increase batch_size as soon as possible

    num_images=5508
    batch_size=51
    imgs = load_test_images(filepath, num_images)
    imgs = preprocess(imgs)
    predicted_masks = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_v_axis = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_h_axis = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_b_axis = np.empty((batch_size,1,img_rows,img_cols))
    for i in range(num_images/batch_size):
        start_time = time.time()
        start = batch_size*i
        end = (batch_size*(i+1))
        print("Generating masks for batch {} ...".format(i))
        img = imgs[start:end]
        predicted_masks = (test(img))
        v_trans = []
        h_trans = []
        b_trans = []        
        for a_img in img:
            x , y , z = augment_data(a_img[0,:,:])
            v_trans.append(x)
            h_trans.append(y)
            b_trans.append(z)
        v_trans = np.array(v_trans).reshape((-1,1,img_rows,img_cols))
        h_trans = np.array(h_trans).reshape((-1,1,img_rows,img_cols))
        b_trans = np.array(b_trans).reshape((-1,1,img_rows,img_cols))
        predicted_masks_v_axis = test(v_trans)
        predicted_masks_h_axis = test(h_trans)
        predicted_masks_b_axis = test(b_trans)
        j=0
        for mask in predicted_masks_v_axis:
            predicted_masks_v_axis[j,0,:,:] = unaugment(mask[0,:,:],1)
            j += 1
        j = 0
        for mask in predicted_masks_h_axis:
            predicted_masks_h_axis[j,0,:,:] = unaugment(mask[0,:,:],2)
            j += 1
        j = 0
        for mask in predicted_masks_b_axis:
            predicted_masks_b_axis[j,0,:,:] = unaugment(mask[0,:,:],3)
            j += 1
        predicted_masks += predicted_masks_v_axis + predicted_masks_h_axis + predicted_masks_b_axis 
        predicted_masks /= 4
        #Compute RLE for the predicted masks and add it to  the list of rle masks
        j = 0
        resizedimages = np.empty((predicted_masks.shape[0],420,580))
        for img in predicted_masks:
            resizedimages[j] = postprocess(img[0])
            j += 1
        for resized_mask in resizedimages:
            rle_mask.append(" ".join(str(r) for r in Rlenc(resized_mask)))
        print("Generating mask for batch {} took {}".format(i,time.time() - start_time))
            
    write_submission_file(num_images)

if __name__== "__main__":
    main()
