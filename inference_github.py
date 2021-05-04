import torch
import scipy.io as sio
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import imageio

from collections import deque
import cv2
from skimage.transform import rescale
import SimpleITK

# from mayavi.core.api import Engine
# from mayavi.sources.vtk_file_reader import VTKFileReader
# from mayavi.modules.surface import Surface
# import mayavi.mlab as mlab

from dataset import *
from model import *# VGG, VGG_whole_average,VGG_whole_attention,VGG_whole_attention_v2, make_layers, make_layers_group_norm, weight_init


def set_model(model_path, batch_size):
    #################################
    ### change the type of model, either baseline or proposed
    #################################
    model = Baseline_vgg(make_layers_instance_norm(), num_classes=[3,3,3], fc_size = 512, device=device).to(device)
    model.load_state_dict(torch.load(model_path))

    model = model.eval()
    return model


def extract_prediction(y1_pred, y2_pred, y3_pred, vol):
    # Procrustes
    sampling_xrange = np.arange(-80,80)
    sampling_yrange = np.arange(-80,80)
    X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
    grid = np.dstack([X, Y])
    grid = np.concatenate((grid,np.zeros([160,160,1])),axis=-1)
    grid += 80
    sample_pt1 = np.array([grid[80,80,0],
                           grid[80,80,1],
                           grid[80,80,2]])
    sample_pt2 = np.array([grid[159,0,0],
                           grid[159,0,1],
                           grid[159,0,2]])
    sample_pt3 = np.array([grid[159,159,0],
                           grid[159,159,1],
                           grid[159,159,2]])
    sample = np.stack((sample_pt1,sample_pt2,sample_pt3), axis=0)
    target = np.stack((y1_pred+80,y2_pred+80,y3_pred+80), axis=0)
    _, tform_pts, tform_parms = procrustes(target, sample, scaling=False, reflection=False)

    # Plane extraction from procrustes parameters
    grid_pred = np.einsum('mni, ij -> jmn', grid, tform_parms['rotation'])
    grid_pred[0,:,:] = grid_pred[0,:,:]+tform_parms['translation'][0]
    grid_pred[1,:,:] = grid_pred[1,:,:]+tform_parms['translation'][1]
    grid_pred[2,:,:] = grid_pred[2,:,:]+tform_parms['translation'][2]
    xx = np.arange(160)
    yy = np.arange(160)
    zz = np.arange(160)
    interp_arr = interpn((xx, yy, zz), vol, np.transpose(grid_pred.reshape((3,160*160))), bounds_error=False, fill_value=0)
    slice_pred = interp_arr.reshape((160,160))
    
    return grid_pred, slice_pred


if __name__ ==  '__main__':
    'Set parameters'
    #################################
    ### edit the parameters in this part
    ################################# 
    video_path = ''
    save_gif = False
    gif_path = ''
    atlas_path = ''
    model_path = ''
    # four corner coordinates for cropping the video
    r1=50  
    r2=850  
    c1=200
    c2=1000
    # flip the video, try both True and False, either one would work
    flip = False
    
    
    
    'Load Video'
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    img_movie = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, img_movie[fc] = cap.read()
        fc += 1
    
    cap.release()
    img_movie_orginal = img_movie.copy()
    
    set_size = img_movie.shape[0]   # can change
    
    
    'Load atlas (or a training volume as atlas) for qualitative comparison'
    mat = sio.loadmat(atlas_path)
    atlas = mat['img_brain']*mat['img_brain_mask']
    
    
    
    'For result visualization'
    fig = plt.figure(figsize=(10, 5))
    ax_img = fig.add_subplot(121)
    ax_atl_pred = fig.add_subplot(122)
    surface_pred = deque([])
    gif_frames = []
    images = deque([])
    
    
    
    'Preprocess Video'
    img_movie = 0.2989*img_movie[:,r1:r2,c1:c2,0] + 0.5870*img_movie[:,r1:r2,c1:c2,1] + 0.1140*img_movie[:,r1:r2,c1:c2,2]   #RGB to grayscale
    rescale_factor = 160/max(img_movie.shape[1], img_movie.shape[2])
    
    for i in range(set_size):
        img_slice = rescale(img_movie[i,:,:], rescale_factor)
        if flip:
            img_slice = np.fliplr(img_slice)
        img_slice = image_normalize(img_slice)

        img_slice = np.expand_dims(img_slice, axis=0)
        images.append(img_slice)
        
    
    
    'Load model'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = set_model(model_path)
    
    
    
    'Prediction'
    with torch.set_grad_enabled(False):
        local_batch = torch.from_numpy(np.asarray(images))
        local_batch = local_batch.to(device=device, dtype=torch.float)
        y1_pred, y2_pred, y3_pred, _ = model(local_batch)
        
        y1_pred = y1_pred.detach().squeeze().cpu().numpy()
        y2_pred = y2_pred.detach().squeeze().cpu().numpy()
        y3_pred = y3_pred.detach().squeeze().cpu().numpy()
    
    
    'Extract slices from atlas for comparison'
    for i in range(set_size):
        _, atl_pred = extract_prediction(y1_pred[i], y2_pred[i], y3_pred[i], atlas)
        img_slice = images[i]
        
        ax_atl_pred.cla()
        ax_atl_pred.imshow(atl_pred, cmap='gray')
        ax_atl_pred.axis('off')
      
        # plot plane
        ax_img.cla()
        ax_img.imshow(np.squeeze(img_slice), cmap='gray')
        ax_img.set_title(i)
        ax_img.axis('off')
        plt.pause(0.01)
    

        # create gif
        if save_gif:
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_frames.append(image)
    
    if save_gif:
        imageio.mimsave(gif_path, gif_frames, fps=8)