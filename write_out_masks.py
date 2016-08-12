import numpy as np
import theano
import theano.tensor as T

import time
import cv2

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, auc

import lasagne


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

def postprocess_img(img):
   
    #img = img.astype('float32')
    #img *= presence_pred_prob
    #img = cv2.threshold(img, 0.4, 1., cv2.THRESH_BINARY)[1]#.astype(np.uint8)
    img = cv2.resize(img, (580, 420))#.astype(np.uint8)*255
    
 
    return img


def postprocess_mask(img):
   
    img = img.astype('float32')
    #img *= presence_pred_prob
    img = cv2.threshold(img, 0.4, 1., cv2.THRESH_BINARY)[1]#.astype(np.uint8)
    img = cv2.resize(img, (580, 420)).astype(np.uint8)*255
    
 
    return img


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
                # trans_imgs, trans_img_masks = augment_data(img,img_mask,patient_idx,image_idx)
                # for trans_img in trans_imgs:
                #     imgs.append(trans_img)
                # for trans_img_mask in trans_img_masks: 
                #     img_masks.append(trans_img_mask)             
                # print "patient {}: loaded image {}".format(patient_idx+1,image_idx+1)
                # prev_img = img
                # prev_mask = img_mask
            except:
                # imgs.append(prev_img)
                # img_masks.append(prev_mask)
                
                # trans_imgs, trans_img_mask = augment_data(prev_img,prev_mask,patient_idx,image_idx) 
                # for trans_img in trans_imgs:
                #     imgs.append(trans_img)
                # for trans_img_mask in trans_img_masks: 
                #     img_masks.append(trans_img_mask) 
                    
                print "patient {}: image or mask {} not found, skipping".format(patient_idx+1,image_idx+1)
          
    print "Augmented Training data stats : {} images have been loaded".format(len(imgs))
    imgs = np.array(imgs).reshape(-1, 1, 420, 580)
    img_masks = np.array(img_masks).reshape(-1, 1, 420, 580)
    return imgs,img_masks

def iterate_minibatches(inputs, targets=None,batchsize=100, shuffle=False):
    if targets is not None:
        assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets is not None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt]


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



def grays_to_RGB(img):
    # Convert a 1-channel grayscale image into 3 channel RGB image
    return np.dstack((img, img, img))


def image_plus_mask(img, mask,channel=0):
    # Returns a copy of the grayscale image, converted to RGB, 
    # and with the edges of the mask added in red

    if img.ndim < 3 or img.shape[2]==1:
        img_color = grays_to_RGB(img)
    else:
        img_color=img 

    print "mask shape: {}".format(mask.shape)
    print "mask data type: {}".format(mask.dtype)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 0  # chan 0 = bright red
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0

    img_color[mask_edges, channel] = 255  # mark the used channel
    return img_color



def augment_data(img):
    
    
    vertical_axis = cv2.flip(img,1)
    print("   Flipped about vertical axis..")
    
    horizontal_axis = cv2.flip(img,0)
    print("   Flipped about Horizontal axis..")
   
    both_axis = cv2.flip(vertical_axis,0)
    print("   Flipped about both axes..")

    return vertical_axis , horizontal_axis , both_axis   

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
 #       print("   Unflipped about vertical axis..")
        return vertical_axis
  
    elif index == 2 :
        horizontal_axis = cv2.flip(img,0)
#        print("   Unflipped about horizontal axis..")
        return horizontal_axis

    elif index == 3:
        h_axis = cv2.flip(img,0)
        v_axis = cv2.flip(h_axis,1)
 #       print("   Unflipped about both axes..")
        return v_axis

    else:
  #      print("   Kept as such")
        return img            



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
    with np.load('./30_epochs_segnet_flip.npz') as f:
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

def test_data(imgs):

    num_images=imgs.shape[0]
    batch_size=51
    imgs = preprocess(imgs)

    print "num images: {}".format(num_images) 

    predicted_masks = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_v_axis = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_h_axis = np.empty((batch_size,1,img_rows,img_cols))
    predicted_masks_b_axis = np.empty((batch_size,1,img_rows,img_cols))
  
    output_mask=[]
    output_probs=[]

    #iterate_minibatches(inputs, targets,batchsize, shuffle=False):
#    for i in range(num_images/batch_size):
    for batch in iterate_minibatches(imgs, None,batch_size, shuffle=False):
        start_time = time.time()
        #start = batch_size*i
        #end = (batch_size*(i+1))
#        print("Generating masks for batch {} ...".format(i))
        imgs = batch#imgs[start:end]
        predicted_masks = (test(imgs))
        v_trans = []
        h_trans = []
        b_trans = []        
        for a_img in imgs:
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
        #resizedimages = np.empty((predicted_masks.shape[0],420,580))
        #resizedimages=[]
        for img in predicted_masks:
            output_probs.append(postprocess_img(img[0]))
            output_mask.append(postprocess_mask(img[0]))
 
 
    return output_mask,output_probs

def plotROC(scores,labels):
    # takes two numpy arrays 

    # flatten out the arrays first
    scores_flat=scores.flatten()
    labels_flat=labels.flatten()
    

    fpr, tpr,thresholds = roc_curve(labels_flat, scores_flat)
    roc_auc = auc(fpr, tpr)


    # now compute tpr+(1-fpr)
    thres_idx=np.argmax(tpr+(1-fpr))
    threshold=thresholds[thres_idx]
    print "thresholds shape {}".format(thresholds.shape)
    print "thresholds: {}".format(thresholds)
    print "threshold index: {}".format(thres_idx)
    print "threshold: {}".format(threshold)
    print "roc auc: {}".format(roc_auc)
  
    return threshold

def thresROC(validation_probs, threshold):


    #img = img.astype('float32')
    #img *= presence_pred_prob
    masks_thresholded=[]
    for img in validation_probs:

        img = cv2.threshold(img, threshold, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)*255
        #img = cv2.resize(img, (580, 420)).astype(np.uint8)*255
        masks_thresholded.append(img)
   
    return masks_thresholded


def main():
  
    #global rle_mask

    data_basepath = '/mnt/storage/users/mfreiber/nerveseg/'
    train_filepath=data_basepath+'train/'
    test_filepath = data_basepath + 'test/'

    num_test_images=51#5508

    #Split into training and validation data

    val_patients = 5

    #imgs= load_test_images(filepath, num_test_images)

#    output_mask=test_data(imgs)
    num_patients=47
    num_images=120
    imgs,img_masks=load_patient_data(train_filepath, num_patients, num_images)

    # normalize image mask data
    #img_masks = img_masks.astype('float32')    
    #img_masks /= 255.


    X_train, X_val = imgs[:-(val_patients*num_images)],imgs[-(val_patients*num_images):]     
    y_train, y_val = img_masks[:-(val_patients*num_images)], img_masks[-(val_patients*num_images):]
  
    y_val_float=y_val.astype('float32')
    y_val_float/= 255.

    validation_mask,validation_probs=test_data(X_val)

    
    threshold = plotROC(np.array(validation_probs).reshape(-1,1,420,580),y_val_float)
        
    validation_mask_roc=thresROC(validation_probs, threshold) 


 #   write_submission_file(num_images)





    pp = PdfPages('./val_masks.pdf');

 

    for (img_idx,img) in enumerate(X_val):
        fig, axis1 =plt.subplots()
#        img=image_plus_mask(img,labels[img_idx],0)

#        print "shape before postprocessing: {}".format(img.shape)
        img=postprocess_img(img[0])

#        print "shape before drawing: {}".format(y_val[img_idx].shape)
#        print "num_validation masks: {}".format(len(validation_mask))
        img=image_plus_mask(img,validation_mask_roc[img_idx],2)
        img=image_plus_mask(img,y_val[img_idx][0],0)
#        print "shape after drawing: {}".format(img)

        # draw labeled mask on image here
        # also draw other predicted mask on image
        # write out the drawnout image
        
     
        plt.imshow(img)
        pp.savefig(fig)




    pp.close()

    print "dice coefficient on validation set: {}".format(dice_coef(y_val_float,np.array(validation_mask_roc).astype('float32')/255.,1.))


#     pp = PdfPages('./masks.pdf');


#     for (img_idx,img) in enumerate(imgs):
#         fig, axis1 =plt.subplots()
# #        img=image_plus_mask(img,labels[img_idx],0)

#         print "shape before postprocessing: {}".format(img.shape)
#         img=postprocess_img(img[0])

#         print "shape before drawing: {}".format(img.shape)
#         img=image_plus_mask(img,output_mask[img_idx],1)


#         # draw labeled mask on image here
#         # also draw other predicted mask on image
#         # write out the drawnout image
        
#         print "shape after drawing: {}".format(output_mask[img_idx].shape)
#         plt.imshow(img)
#         pp.savefig(fig)




#     pp.close()



if __name__== "__main__":
    main()

