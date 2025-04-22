# cone-beam tomography, lite version

# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw



import numpy as np

import time

import copy 

import astra

import matplotlib.pyplot as plt 

from skimage.transform import rescale, resize

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import mean_squared_error as mse



def normalizeRange(A, source_min, source_d, target_min = 0.0, target_d = 1.0): 

    B = (A - source_min) / source_d * target_d + target_min

    return B    



def normalize(A, minimum = 0.0, maximum = 1.0): 

    mini = np.min(A)

    maxi = np.max(A)

    return normalizeRange(A, mini, maxi - mini, minimum, maximum - minimum)



def evaluateVolume(vol, target, data_range = None):

    diff = vol - target

    v_mae = np.mean(np.abs(diff))

    v_mse = mse(vol, target) 

    v_ssim = 0.0

    v_psnr = 0.0

    if data_range is None:

        data_range = target.max() - target.min()    

    v_ssim = ssim(vol, target, data_range = data_range) 

    v_psnr = psnr(vol, target, data_range = data_range)        

    print(' ', v_mae, v_mse, v_ssim, v_psnr)
    
    return v_mae, v_mse, v_ssim, v_psnr       



def showImageTable(images, rows, cols, size = None, figsize = None, cmap = 'gray', caption = None):

    fig = plt.figure(figsize=figsize)

    for i, image in enumerate(images):  

        ax = fig.add_subplot(rows, cols, i + 1)

        if not (caption is None):

            ax.title.set_text(caption[i])

        if size is not None:

            image = np.reshape(image, size)

        if len(image.shape) > 2:

            nH, nW, nC = image.shape

            if nC == 1:

                image = np.reshape(image, image.shape[0:2])

        else:

            nH, nW = image.shape

            nC = 1        

        if nC == 1:     

            plt.imshow(image, cmap=cmap, vmin = 0, vmax = 1)

        else:

            plt.imshow(image, vmin = 0, vmax = 1)

    plt.show()  



# ----------------------------------------------------

# Projection shapes: (detector_rows, angles, detector_columns)

class ConeRec:

    def __init__(self, vol_shape, proj_shape, scan_range = (0, 2 * np.pi), angles = None, volume = None, proj = None,

                 det_width = 1.0, det_height = None, source_origin = 640., origin_det = 384.,

                 algo = 'FDK_CUDA', iterations = 1000):

        self.vol_shape = vol_shape      # (d, h, w)

        self.depth, self.height, self.width = self.vol_shape

        self.proj_shape = proj_shape    # (slices, angles, detectors)

        self.n_det_rows, self.n_angles, self.n_det_cols = self.proj_shape

        self.scan_range = scan_range

        self.proj_mode = 'cone'      

        # create_vol_geom(Y, X, Z)``:  

        self.vol_geom = astra.create_vol_geom(self.height, self.width, self.depth)

        self.vol_id = astra.data3d.create('-vol', self.vol_geom, data = volume)

        if angles is None:

            self.angles = np.linspace(self.scan_range[0], self.scan_range[1], self.n_angles, False)

        else:

            self.angles = angles

        self.det_width = det_width

        self.det_height = self.det_width if det_height is None else det_height

        self.source_origin = source_origin

        self.origin_det = origin_det



        # create_proj_geom('cone', detector_spacing_x, detector_spacing_y, det_row_count, det_col_count, angles, source_origin, source_det)

        self.proj_geom = astra.create_proj_geom(self.proj_mode, 

                                                self.det_width, self.det_height,

                                                self.n_det_rows, self.n_det_cols,

                                                self.angles, 

                                                self.source_origin,

                                                self.origin_det)  

        self.proj_id   = astra.data3d.create('-sino', self.proj_geom, data = proj)



        # Available algorithms:

        # 'FDK_CUDA', 'SIRT3D_CUDA', 'CGLS3D_CUDA'

        self.algo = algo   

        self.iterations = iterations          

        self.alg_cfg = astra.astra_dict(self.algo)

        self.alg_cfg['ProjectionDataId'] = self.proj_id

        self.alg_cfg['ReconstructionDataId'] = self.vol_id

        self.alg_id = astra.algorithm.create(self.alg_cfg)  

        



    @classmethod

    def createf(cls, vol_size, n_angles = 720, algo = 'FDK_CUDA', iterations = 1000):

        ang_range = np.pi * 2       

        vol_shape = (vol_size, vol_size, vol_size)

        det_row_count = int(vol_size * 2)  

        det_col_count = int(vol_size * 2)   

        source_origin = int(vol_size * 2.05)

        origin_det = int(vol_size * 1.)

        return cls(vol_shape = vol_shape, proj_shape = (det_row_count, n_angles, det_col_count), 

                 scan_range = (0, ang_range), 

                 angles = np.linspace(0,  ang_range, num = n_angles, endpoint = False), 

                 det_width = 1.0, source_origin = source_origin, origin_det = origin_det,

                 algo = algo, iterations = iterations)



    def create512f(cls, n_angles = 720, algo = 'FDK_CUDA', iterations = 1000):

        return cls.createf(vol_size = 512, n_angles = n_angles, algo = algo, iterations = iterations)

    @classmethod

    def create256f(cls, n_angles = 360, algo = 'FDK_CUDA', iterations = 1000):

        return cls.createf(vol_size = 256, n_angles = n_angles, algo = algo, iterations = iterations)                

    @classmethod

    def create128f(cls, n_angles = 180, algo = 'FDK_CUDA', iterations = 1000):

        return cls.createf(vol_size = 128, n_angles = n_angles, algo = algo, iterations = iterations)



    def project(self, volume = None, keep_id = False):

        if not (volume is None):

            astra.data3d.store(self.vol_id, volume)  

        proj_id, proj = astra.creators.create_sino3d_gpu(self.vol_id, self.proj_geom, self.vol_geom)

        if keep_id:

            return proj_id, proj

        else:

            proj = np.array(proj) # (slices, angles, detectors)

            astra.data3d.delete(proj_id)

            return proj



    # Projection shapes: (slices, angles, detectors)

    def reconstruct(self, proj = None):

        if not(proj is None):       

            astra.data3d.store(self.proj_id, proj)       

        astra.algorithm.run(self.alg_id, self.iterations)

        rec = astra.data3d.get(self.vol_id)   

        return rec



    # Destructor

    def release(self):

        astra.data3d.delete(self.vol_id)

        astra.data3d.delete(self.proj_id)

        astra.algorithm.delete(self.alg_id) 
        
