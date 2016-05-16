import sys 
import glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt  
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import square
from skimage.segmentation import find_boundaries as fbn
from scipy.ndimage.morphology import binary_fill_holes



sys.dont_write_bytecode = True
plt.xkcd(scale=.5, length=100, randomness=2)


class GenArray:
    def __init__(self, directory, start = 1, ends = 100, 
                 xrng = (0,512) , yrng = (0,512)):
        self.directory = directory
        self.start = start
        self.ends = ends
        self.xrng = xrng 
        self.yrng = yrng 
        self.gen()
        
    def gen(self):
        """
        reads files from given directory, then returns a 3D array. 
        This array is somehow like a HU(x, y, z) function. every spacial
        coordinate has a specific HU value, so we can access to it's 
        grayscale, and apply image processing functions and so forth. 
        """
        #pre-allocating
        self.arr = np.zeros((self.xrng[1] - self.xrng[0],  # x - dimension
                             self.yrng[1] - self.yrng[0],  # y - dimension
                             self.ends - self.start))      # z - dimension
        
        cnt = 0
        for i in range(self.start,self.ends):
            # a temporary variable
            tmp_dcm = dicom.read_file(self.directory[i])
            # assigning values to pre-allocated array
            self.arr[:,:,cnt] = tmp_dcm.pixel_array[self.xrng[0]:self.xrng[1],
                                                    self.yrng[0]:self.yrng[1]]
            cnt += 1 

    def size(self):
        return self.arr.shape


class Segment(GenArray):
    def __init__(self, GenArrayObj,
                 sigma = 2, low_threshold=200):
        
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.arr = GenArrayObj.arr
        self.size = GenArrayObj.size()
        self.gen_mask()
        
    def seg_sect(self, img):
        img_canny = canny(img, sigma=self.sigma,
                          low_threshold=self.low_threshold)
        
        img_dilate = binary_dilation(img_canny, square(3))
        img_erode = binary_erosion(img_dilate, square(3))
        img_fill = binary_fill_holes(img_erode)
        
        return img_fill
    
    def gen_mask(self):
        slices = self.size[-1]
        
        # pre-allocating mask
        self.mask = np.zeros_like(self.arr, dtype=np.bool)
        
        for i in range(slices):
            self.mask[:, :, i] = self.seg_sect(self.arr[:, :, i])



def read_files(path, extension = '*.dcm'):
    """ Read files with specific extension in a dirctory, """

    path += '/' if not path[-1] == '/' else ''
    directory = path + extension

    # finds all files with specific format, 
    # here, we are a bit more interested in dicom files

    files = glob.glob(directory)        
    files.sort()

    return files


def bin_boundary(arr, method='surf'): 
    """
        calculate a new set of points are located in B array 
        this set only containts boundary points (voxels) 
        and inner voxels are omited 
        using BB instead of B can decrease cost of other functions and 
        algorithms in this notebook 
        
        Input : 
            arr:  a 3D binary array 
            method : 'surf' generate a closed surface (default) and 
                     'cont' genr8s contours 
                     
        Output : 
            BB: a 3*n array contain location of each point [x,y,z]
    """
    B = deepcopy(arr)
    if method == 'surf': 
        for lyr in range(B.shape[0]-1):
            B[lyr] = np.logical_or(np.logical_and(np.logical_xor(B[lyr],B[lyr+1])
                                                ,B[lyr])
                                 ,fbn(B[lyr]))
        return B
    
    elif method == 'cont':
        for lyr in range(B.shape[0]-1):
            B[lyr] = fbn(B[lyr])
        return B


# calculate slope and interception after calibration
# nevermind, not in our scope
phantom_slope = 1
phantom_intercept = -1023

# functions : 

rho_KHP = lambda HU: phantom_slope * HU + phantom_intercept
rho_ash = lambda rho_KHP : 1.22 * rho_KHP / 1000. + 0.0526

# Elastic modulus    
E_modul = lambda r: (np.logical_and(r>0, r<=0.27) * 33900*r)**2.20 + \
            ((r>=0.60) * 10200*r)**2.01 + \
            (np.logical_and(r>0.27, r<0.60) * (5307*r + 469)) + (r<0) * 15000

