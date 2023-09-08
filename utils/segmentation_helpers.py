#Helper function for segmentation pipeline

import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from scipy import ndimage
from PIL import Image, ImageChops
import skimage.measure as measure 
import imutils
from tqdm import tqdm
from skimage import color
from skimage.segmentation import clear_border
import matplotlib.patches as patches
from skimage.transform import downscale_local_mean


'''Multiscale template matching:
Function to match the raman image(100x100) with the 
bright field image(2048x2048) to find the raman field of view

:params:
img : bright field image of control beads
temp : Peak 669 of the raman dat file of control beads

:return:
location of the raman image on the bright field image and 
dimensions raman image should take to match with the raman
field of view. 

'''
def match_temp(img,temp):
    t_dim = np.shape(temp)[0]
    i_dim = np.shape(img)[0]
    matched_dims = None
    if(len(np.shape(img))>2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert image to grayscale
    if(len(np.shape(temp))>2):
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)       #Convert template to grayscale
    img = cv2.Canny(img, 15, 100)                         #edge detection in image
    for dims in tqdm(range(t_dim,i_dim)):
        resized = imutils.resize(temp, width = dims, height = dims)
        edged = cv2.Canny(resized,50,200)                   #edge detection in template 
        match = cv2.matchTemplate(img,resized, cv2.TM_CCOEFF)
        (_, maxVal,_,maxLoc) = cv2.minMaxLoc(match)
        if matched_dims==None or maxVal>matched_dims[0]:
            matched_dims = (maxVal,maxLoc,dims)
    return matched_dims 



'''Function to downsample an Image:

:params: 
img: Image 
size: Desired size of the image smaller than size of Image 

:return:
Image of dimensions equal to size(param) 

'''
def downsample(img,size):
    factor = img.shape[0]/size
    small = cv2.resize(img, (int(img.shape[1]//factor), 
                             int(img.shape[0]//factor)),
                            interpolation=cv2.INTER_NEAREST)
    small = (np.array(small)).astype('int')
    return small

'''Function to crop raman field of view
:params:
mask: Image to be cropped
loc: returned from template matching 
dims: returned from template matching

:return:
Raman field of view cropped from image
'''

def crop1d(mask,loc,dims):
    return mask[loc[0]:loc[0]+dims,loc[1]:loc[1]+dims]


'''Funtion to crop raman field of view from a list of masks'''

def crop2d(masks,loc,dims):
    cropped = []
    for mask in masks:
        cropped.append(mask[loc[0]:loc[0]+dims,loc[1]:loc[1]+dims])
    return cropped
        

'''Function to upsample:

:params: 
img: Image 
size: Desired size of the image bigger than the size of the Image

:return:
Image of dimensions equal to size(param) 

'''
def upsample(img,size):
    factor = size/img.shape[0]
    big = cv2.resize(img, (int(img.shape[1]*factor), 
                             int(img.shape[0]*factor)),
                            interpolation=cv2.INTER_NEAREST)
    big = (np.array(big)).astype('int')
    return big


'''Check calibration

:param:
bf : brightfield image
ram : raman image 
loc, dims: location and dimensions of the raman image wrt brightfield
image from template matching 
alpha: transparency

:return:
plot of raman overlayed on bright field to check calibration 

'''
def check_calib(bf,ram,loc,dims,alpha):
    l = loc[0]
    r = bf.shape[0]-(loc[0]+dims)
    t = loc[1]
    b = bf.shape[0]-(loc[1]+dims)
    new_ram = cv2.copyMakeBorder(ram, t, b, l, r, cv2.BORDER_CONSTANT,value=255)
    plt.figure(figsize=(8,8))
    plt.imshow(bf)
    plt.imshow(new_ram,cmap='coolwarm',alpha=alpha)



'''Plot all images for visualization

:param:
names: titles for images
images: Images to plot

:return:
Grid of plots 

'''
def plot_all(names,images,rows,columns):
    fig,ax = plt.subplots(rows,columns,figsize=(25,25))
    for a,name,img in zip(ax.flat,names,images):
        a.imshow(img)
        a.set_title(name)
        [a.axis('off') for a in ax.flat]
    plt.tight_layout()

    
    
'''Function to remove small cells of a particular area
:param:
masks: segmentation masks 
area: threshold area 

:return:
filtered_masks without cells of area less than input area
'''
def remove_small(masks,area):
    filtered_masks=[]
    for m in masks:
        small_labels=[]
        temp = np.copy(m)
        props = measure.regionprops(m)
        for prop in props:
            if(prop.area<area):
                small_labels.append(prop.label) 
        for l in small_labels:
            temp[temp==l]=0
        filtered_masks.append(temp)
    return filtered_masks


'''Function to remove cells from along the border
:param:
masks: segmentation masks

:return:
filtered_masks without partial cells along the borders
'''

def remove_border(masks):
    filtered_masks = []
    for m in masks:
        fm = clear_border(m)
        filtered_masks.append(fm)
    return filtered_masks

'''Function to check if cells greaters than certain area value are actually cells 
:param:
masks: segmentation masks
area: threshold area

:return:
plots showing segmentation masks with big cells(area_of_cell>area) identified
'''
def validate_big_cells(masks,area,rows,columns):
    fig,axs=plt.subplots(rows,columns,figsize=(25,25))
    for m,a in zip(masks,axs.flat):
        props = measure.regionprops(m)
        box = []
        [box.append(prop.bbox) for prop in props if prop.area>=area]
        rectangles = []
        for bbox in box:
            width = bbox[3]-bbox[1]
            height = bbox[2]-bbox[0]
            rect = patches.Rectangle((bbox[1],bbox[0]),width,height,linewidth=1,edgecolor='r',facecolor='none')
            rectangles.append(rect)
        a.imshow(m)
        [a.add_patch(rect) for rect in rectangles]
        [a.axis('off') for a in axs.flat]
        
'''Function to find pixel positions of cells 
:param:
masks: segmentation masks

:return:
label_pos: pixel positions of cells within each mask
'''
def cell_positions(masks):
    label_pos = []
    for mask in masks:
        labels = np.unique(mask)
        labels = labels[labels!=0]
        label_pos.append([np.transpose(np.where(mask == label)) for label in labels])
    return label_pos


'''Function to bin an image
:param:
img: image

:return:
binned image
'''
def bin_data(img,bin_size):
    chans = len(np.shape(img))
    if(chans==3):
        return downscale_local_mean(img,(bin_size,bin_size,1))
    else:
        return downscale_local_mean(img,(bin_size,bin_size))
