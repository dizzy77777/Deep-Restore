import numpy as np
from skimage import color
from image_utils import split_img
from imageio import imread, imwrite

def read_masks(mask_list, size, roi, padding_width):
    n = 0
    masks = np.zeros((len(mask_list)*size*size, roi+2*padding_width, roi+2*padding_width))
    for i in mask_list:
        '''
        img = color.rgb2gray(imread(i)) / 255.
        
        class_1 = np.where(img[:roi, :roi] == 0)
        class_2 = np.where(img[:roi, :roi] == 1)

        masks_train[j,class_1[0],class_1[1],0] = 1
        masks_train[j,class_2[0],class_2[1],1] = 1
        '''
        
        ### just one part of image
        '''
        img = color.rgb2gray(imread(i)) / 255.
        masks_train[j,:,:] = ((img[:roi, :roi]).astype(np.int)).copy()
        '''
        
        
        ### each split of image
        img = color.rgb2gray(imread(i)) / 255.
        img_split = split_img(img,size,roi)      
        
        
        for j in range(size*size):
            masks[n*size*size + j,:,:] = np.pad(img_split[j,:,:],padding_width,'reflect').copy()
                        
        n = n + 1
    return masks


def read_inputs(inputs_list, size, roi, padding_width):
    n = 0
    inputs = np.zeros((len(inputs_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    for i in inputs_list:
        img = imread(i).astype(float) 
        if(np.max(img) > 255.):
            img /= 65535.0
        else:
            img /= 255.0
            
        ### each split of image
        img_split = split_img(img,size,roi)
        
        for j in range(size*size):

            inputs[n*size*size + j,:,:] = np.pad(img_split[j,:,:],((padding_width,padding_width),(padding_width,padding_width),(0,0)),'reflect').copy()
        
        n = n + 1
    return inputs
   
