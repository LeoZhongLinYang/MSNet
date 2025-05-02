import os
import numpy as np
from cone import ConeRec, normalize, showImageTable, evaluateVolume # import the lite version
import time
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage.transform import rescale, resize

vol_id_list = ['N005', 'N012', 'N300', 'C002', 'C004', 'C012', 'L004', 'L006', 'L014', 'L019']

for vol_id in vol_id_list:
    vol_path = f'/media/jamescheng/My Passport/LDCT/volume/{vol_id}.npy'
    rec_path = f'/media/jamescheng/My Passport/LDCT/result/image_hdnet+/result/{vol_id}.npy_hdnet+_image.npy'

    rec = np.load(rec_path)
    vol = np.load(vol_path) 
    
    ####################################
    rec = np.clip(rec, 0.0, None)
    rec = normalize(rec)    
    vol = normalize(vol)    
    ####################################

    print(vol_id, end="") 
    evaluateVolume(rec, vol)
