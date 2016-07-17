import sys
sys.settrace
import glob
import os
import os.path
import re

import cv2

import scipy.spatial.distance
import numpy as np
import collections
import itertools
import string


IMAGE_DIR        = '../nerve_data/train/'
AUG_IMAGE_DIR = '../nerve_data/train_augmented/'
TILE_MIN_SIDE    = 50     # pixels; see tile_features()
MIN_MATCH_COUNT = 50  

def estimate_homography(img_1,img_2):

    sift = cv2.xfeatures2d.SIFT_create()

    kp_1, des_1 = sift.detectAndCompute(img_1,None)
    kp_2, des_2 = sift.detectAndCompute(img_2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    import math
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print "des1: {}".format(len(des_1))
    # print "des2: {}".format(len(des_2))
    if len(des_1)>MIN_MATCH_COUNT and len(des_2)> MIN_MATCH_COUNT: #and ( math.fabs(len(des_1)-len(des_2))<150): large differences in number of features create segfaults in the flann matcher.. 
        matches = flann.knnMatch(des_1,des_2,k=2)
    else:
        matches = []
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    print "length of good: {}".format(len(good))
     
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #mask gives the correspondences used for the homography

        return M

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        return None

def image_seq_start(dists, f_start):

    # Given a starting image (i.e. named f_start), greedily pick a sequence 
    # of nearest-neighbor images until there are no more unpicked images. 

    f_picked = [f_start]
    f_unpicked = set(dists.keys()) - set([f_start])
    f_current = f_start
    dist_tot = 0

    while f_unpicked:

        # Collect the distances from the current image to the 
        # remaining unpicked images, then pick the nearest one 
        candidates = [(dists[f_current][f_next], f_next) for f_next in f_unpicked]
        dist_nearest, f_nearest = list(sorted(candidates))[0]

        # Update the image accounting & make the nearest image the current image 
        f_unpicked.remove(f_nearest)
        f_picked.append(f_nearest)
        dist_tot += dist_nearest
        f_current = f_nearest 

    return (dist_tot, f_picked)



def image_sequence(dists):

    # Return a sequence of images that minimizes the sum of 
    # inter-image distances. This function relies on image_seq_start(), 
    # which requires an arbitray starting image. 
    # In order to find an even lower-cost sequence, this function
    # tries all possible staring images and returns the best result.

    f_starts = dists.keys()
    seqs = [image_seq_start(dists, f_start) for f_start in f_starts]
    dist_best, seq_best = list(sorted(seqs))[0]
    return seq_best


def feature_dist(feats_0, feats_1):
    # Definition of the distance metric between image features
    return scipy.spatial.distance.euclidean(feats_0, feats_1)


def feature_dists(features):
    # Calculate the distance between all pairs of images (using their features)
    dists = collections.defaultdict(dict)
    f_img_features = features.keys()
    for f_img0, f_img1 in itertools.permutations(f_img_features, 2):
        dists[f_img0][f_img1] = feature_dist(features[f_img0], features[f_img1])
    return dists




def tile_features(tile, tile_min_side = TILE_MIN_SIDE):
    # Recursively split a tile (image) into quadrants, down to a minimum 
    # tile size, then return flat array of the mean brightness in those tiles.
    tile_x, tile_y = tile.shape
    mid_x = tile_x / 2
    mid_y = tile_y / 2
    if (mid_x < tile_min_side) or (mid_y < tile_min_side):
        return np.array([tile.mean()]) # hit minimum tile size
    else:
        tiles = [ tile[:mid_x, :mid_y ], tile[mid_x:, :mid_y ], 
                  tile[:mid_x , mid_y:], tile[mid_x:,  mid_y:] ] 
        features = [tile_features(t) for t in tiles]
        return np.array(features).flatten()

def image_features(img):
    return tile_features(img)   # a tile is just an image...


def to_mask_path(f_image):
    # Convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(f_image)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)



def grays_to_RGB(img):
    # Convert a 1-channel grayscale image into 3 channel RGB image
    return np.dstack((img, img, img))



def image_plus_mask(img, mask):
    # Returns a copy of the grayscale image, converted to RGB, 
    # and with the edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # chan 0 = bright red
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    return img_color


def add_masks(images):
    # Return copies of the group of images with mask outlines added
    # Images are stored as dict[filepath], output is also dict[filepath]
    images_plus_masks = {} 
    for f_image in images:
        img  = images[f_image]
        mask = cv2.imread(to_mask_path(f_image))
        images_plus_masks[f_image] = image_plus_mask(img, mask)
    return images_plus_masks


def get_image(f):
    # Read image file 
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print 'Read:', f
    return img


def get_patient_images(patient):
    # Return a dict of patient images, i.e. dict[filepath]
    f_path = IMAGE_DIR + '%i_*.tif' % patient 
    f_ultrasounds = [f for f in glob.glob(f_path) if 'mask' not in f]
    images = {f:get_image(f) for f in f_ultrasounds}
    return images


def image_sequence(dists):

    # Return a sequence of images that minimizes the sum of 
    # inter-image distances. This function relies on image_seq_start(), 
    # which requires an arbitray starting image. 
    # In order to find an even lower-cost sequence, this function
    # tries all possible staring images and returns the best result.

    f_starts = dists.keys()
    seqs = [image_seq_start(dists, f_start) for f_start in f_starts]
    dist_best, seq_best = list(sorted(seqs))[0]
    return seq_best

def extract_indices(image_filename):
    #extracts patient number and image number from a filename 

    pattern = re.compile(r"""(?P<patient_number>.*?) # patient number
                             \_                      # underscore
                             (?P<image_number>.*?)   # image number
                             \.tif""", re.VERBOSE)


    match = pattern.match(os.path.basename(image_filename))
    patient_number=match.group("patient_number")
    image_number=match.group("image_number")
    return patient_number,image_number


def mask_from_image(image_filename):

    patient_number,image_number=extract_indices(image_filename)
    filename_out="{}_{}_mask.tif".format(patient_number,image_number)
    print "filename out: {}".format(os.path.dirname(image_filename)+filename_out)
    return os.path.dirname(image_filename)+'/'+filename_out 


def aug_mask_from_image(image_filename):

    patient_number,image_number=extract_indices(image_filename)
    filename_out="{}_{}_mask_augmented.tif".format(patient_number,image_number)
    return AUG_IMAGE_DIR+'/'+filename_out 

def main():
    

    patient_idx=1    

    img_batch={}
    images = get_patient_images(patient=patient_idx)
    images_masks = add_masks(images)
    features     = { f : image_features(images[f]) for f in images }
    dists        = feature_dists(features)
    filename_sequence        = image_sequence(dists)


    previous_image=cv2.imread(filename_sequence[0])
    previous_mask=cv2.imread(mask_from_image(filename_sequence[0]))


    for current_image_index in range(1,len(filename_sequence)):

        current_image=cv2.imread(filename_sequence[current_image_index])
        current_mask=cv2.imread(mask_from_image(filename_sequence[current_image_index]))
        H=estimate_homography(previous_image,current_image)

        h,w,c=previous_image.shape

        if H is not None: 

            if np.sum(current_mask)==0:

                current_mask=cv2.warpPerspective(previous_mask, H,(w,h))#[, dst[, flags[, borderMode[, borderValue]]]])
                print "Warped image!"

        previous_image=current_image
        previous_mask=current_mask
        # write out the augmented mask
        img_batch[aug_mask_from_image(filename_sequence[current_image_index])]= current_mask


    for filename,mask in img_batch.iteritems():
        cv2.imwrite(filename,mask)

main()

