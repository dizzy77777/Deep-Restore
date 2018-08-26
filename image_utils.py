
import numpy as np

def split_img(img, size, roi):
    # splits 2d image into size*size equal parts, that are roi*roi large
    if size == 1:
        if len(img.shape) == 2:
                extract = np.zeros((1, roi, roi))
                extract[0,:,:] = img
                return extract
        else:
                extract = np.zeros((size*size, roi, roi, 3))
                extract[0,:,:,:] = img
                return extract

    if len(img.shape) == 2:
        vspl = np.vsplit(img, size)
    
        extract = np.zeros((size*size, roi, roi))
    
        for i in range (size):
            hspl = np.hsplit(vspl[i], size)
            for j in range(size):
                extract[i*size + j,:,:] = hspl[j].copy()

        return extract
    else:
        vspl = np.vsplit(img, size)
        
        extract = np.zeros((size*size, roi, roi, 3))
        
        for i in range (size):
            hspl = np.hsplit(vspl[i], size)
            for j in range(size):
                extract[i*size + j,:,:,:] = hspl[j].copy()

        return extract
    
def combine_img(splits, size, roi):
    # combine splits of size*size equal parts, that are roi*roi large into 2d image
    if size == 1:
        if len(splits.shape) == 3:
                img = np.zeros((roi*size, roi*size))
                img[:,:] = splits
                return img
        else:
                img = np.zeros((roi*size, roi*size, 3))
                img[:,:,:] = splits
                return img
    # size*size should be splits.shape[0]
    if len(splits.shape) == 3:
        img = np.zeros((roi*size, roi*size))
        for i in range(size):
            for j in range(size):
                img[i*roi:(i+1)*roi, j*roi:(j+1)*roi] = splits[j+i*size]
        return img
    else:
        img = np.zeros((roi*size, roi*size, 3))
        for i in range(size):
            for j in range(size):
                img[i*roi:(i+1)*roi, j*roi:(j+1)*roi] = splits[j+i*size]
        return img

def threshold_mean(lists, thresh1, thresh2):
    # just take images where the mean value is larger than thresh
    
    means = np.mean(lists, axis=((1,2)))
    #print(means)
    #print(np.mean(means))
    ind = np.where((means > thresh1) & (means < thresh2))
    
    return ind

def mirror_combine_data(in_data):
    
    flip_data = in_data[:,:,::-1]
    
    return np.vstack((in_data, flip_data))

def reverse_flow_combine_data(in_data):
    
    reverse_flow = np.concatenate((in_data[:,:,:,6:9], in_data[:,:,:,3:6], in_data[:,:,:,0:3]), axis = 3)

    return np.vstack((in_data, reverse_flow))


def generate_set_from_idx(full_data_set, ind, size):
    if len(full_data_set.shape) == 4:
        gen = np.zeros((ind.shape[0]*(size**2), full_data_set.shape[1], full_data_set.shape[2], full_data_set.shape[3]))
    elif len(full_data_set.shape) == 3:
        gen = np.zeros((ind.shape[0]*(size**2), full_data_set.shape[1], full_data_set.shape[2]))
    
    for i, idx in enumerate(ind):       
        gen[i*(size**2):(i+1)*(size**2)] = full_data_set[idx*(size**2):(idx+1)*(size**2)]
    return gen