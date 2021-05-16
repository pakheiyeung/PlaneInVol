'''
Part of the code is adapted from Christoph Gohlke's code (https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
'''
from torch.utils import data
import scipy.io as sio
import numpy as np
from scipy.interpolate import interpn
import math
import itertools
import random
from random import uniform
from imgaug import augmenters as iaa
import cv2
from skimage.morphology import convex_hull_image
import sys
import scipy
from scipy import ndimage
import scipy.misc
from skimage.transform import resize


def find_norm_vector(azimuth, elevation):
#    azimuth = azimuth*2*math.pi/360
#    elevation = elevation*2*math.pi/360

    a = np.cos(elevation)*np.cos(azimuth)
    b = np.cos(elevation)*np.sin(azimuth)
    c = np.sin(elevation)
    #d = r*((a**2+b**2+c**2)**0.5)-(a*80+b*80+c*80)

    return np.array([a,b,c])

def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R = np.zeros((3,3))
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

    return R

def EulerToRotation_Z(angle):
    angle = angle*2*math.pi/360
    ca = np.cos(angle)
    sa = np.sin(angle)

    R = np.zeros((3,3))
    R[0,0] = ca
    R[0,1] = -sa
    R[1,0] = sa
    R[1,1] = ca
    R[2,2] = 1.0

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])#*360/2/math.pi

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data



def image_normalize(image):
    image_normalized = image/127.5-1

#    image_non_zero = image[np.nonzero(image)]
#    image_mean = np.mean(image_non_zero)
#    image_std = np.std(image_non_zero)
#
#    image_normalized = (image-image_mean)/image_std
#    image_normalized[image==0] = 0

    return image_normalized

def procrustes(X, Y, scaling=True, reflection=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def grid_translation(grid, x, y, z):
    grid_copy = grid.copy()
    direction_row = grid[:,0,0]-grid[:,0,159]
    direction_col = grid[:,0,0]-grid[:,159,0]
    normal = np.cross(direction_row, direction_col)

    direction_row = unit_vector(direction_row)*x
    direction_col = unit_vector(direction_col)*y
    normal = unit_vector(normal)*z

    movement = direction_row+direction_col+normal
    grid_copy[0,:,:] = grid_copy[0,:,:]+movement[0]
    grid_copy[1,:,:] = grid_copy[1,:,:]+movement[1]
    grid_copy[2,:,:] = grid_copy[2,:,:]+movement[2]

    return grid_copy

def Sampling_grid(azimuth, elevation, r_z, t_x, t_y, t_z, rotation_random):
    # Normal vector of the plane
    norm_vector = find_norm_vector(azimuth, elevation)
    norm_vector = norm_vector/np.linalg.norm(norm_vector)
    norm_vector = np.matmul(rotation_random, norm_vector)
    norm_vector[2]=abs(norm_vector[2])

    # Rotation matrix
    rotation_z = EulerToRotation_Z(r_z)
    rotation_p = R_2vect(np.array([0,0,1]), norm_vector)
    rotation = np.matmul(rotation_p,rotation_z)
#    rotation = np.matmul(rotation_random,rotation)

    # euler angle
    #euler = rotationMatrixToEulerAngles(rotation)
    angle_xy = np.array([azimuth, elevation])
    angle_z = np.array([r_z*2*math.pi/360])
    
    # Grid for sampling
    sampling_xrange = np.arange(-80,80)
    sampling_yrange = np.arange(-80,80)
    X, Y = np.meshgrid(sampling_xrange, sampling_yrange)
    grid = np.dstack([X, Y])
    grid = np.concatenate((grid,np.zeros([160,160,1])),axis=-1)
    grid_rot = np.einsum('ji, mni -> jmn', rotation, grid)
    grid_rot = grid_translation(grid_rot, 0, 0, t_z)
    
    return grid_rot, angle_xy, angle_z

def flood_fill_hull(image):
    image = image.copy().astype(int)    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



def find_dist(mask_vol, azimuth, elevation, rotation_random):
    '''
    find max distance until very little content is observed (near or out of the skull)
    '''
    
    xx = np.arange(160)
    yy = np.arange(160)
    zz = np.arange(160)
    temp_store={}
    
    def find_area(a, e, current_d):
        grid,_,_ = Sampling_grid(a, e, 0, 0, 0, current_d, rotation_random)
        grid+=80
        
        interp_arr = interpn((xx, yy, zz), mask_vol, np.transpose(grid.reshape((3,160*160))), bounds_error=False, fill_value=0)
        mask_slice = interp_arr.reshape((160,160))
        
        temp_area = np.count_nonzero(mask_slice)/1000
        return temp_area
    
    for i in range(len(azimuth)):
        for dist in range(15,80,5):#[40,50,55,60,65,70,75]:
            current_area = find_area(azimuth[i], elevation[i], dist)
            
            if current_area<4:
                final_distance_positive=dist
#                d = dist
#                while True:
#                    d-=1
#                    current_area = find_area(azimuth[i], elevation[i], d)
#                    if current_area>=7:
#                        final_distance_positive=d+1
#                        break
                break
        for dist in range(-15,-80,-5):#[-40,-45,-50,-55,-60,-65,-70,-75]:
            current_area = find_area(azimuth[i], elevation[i], dist)
            if current_area<4:
                final_distance_negative=dist
#                d = dist
#                while True:
#                    d+=1
#                    current_area = find_area(azimuth[i], elevation[i], d)
#                    if current_area>=7:
#                        final_distance_negative=d-1
#                        break
                break
        
        temp_store.update({azimuth[i]:[final_distance_positive, final_distance_negative]})
    
    return temp_store
        


class Dataset(data.Dataset):
    def __init__(self, image_list, **params):
        'Initialization'
        self.batch_size = params['batch_size']
        self.mode = params['mode']
        self.sample_num = params['sample_num'] # number of normal per volume, 50
        self.dist_num = params['dist_num'] # number of slice per normal, 15
        self.create_list_and_ID(image_list)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Load data'
        ID = self.list_IDs[index]
        img_vol = self.vol_store[self.info[ID]['file']]
        param_ids = self.info[ID]['param_ids']
        images = np.empty((self.batch_size,1,160,160))
        y1 = np.zeros((self.batch_size,3))
        y2 = np.zeros((self.batch_size,3))
        y3 = np.zeros((self.batch_size,3))
        
        for count, num in enumerate(param_ids):
            'Get the parameters for sampling'
            plane_info = self.sampling_params[num]
            a = plane_info[-2]#+random.uniform(0,2*math.pi/((np.sqrt(5)+1)/2))
            e = plane_info[-1]
            r = plane_info[0]+random.uniform(0,90)
            dist_interval = (self.dist_store[self.info[ID]['file']][a][0]+self.dist_store[self.info[ID]['file']][a][1])/(self.dist_num-1)/2
            z = plane_info[1]*(self.dist_store[self.info[ID]['file']][a][0]-self.dist_store[self.info[ID]['file']][a][1])+self.dist_store[self.info[ID]['file']][a][1]+random.uniform(-dist_interval,dist_interval)
            
            val = abs(z)
            x = random.uniform(-min(20,val),min(20,val))
            y = random.uniform(-min(20,val),min(20,val))

    
            sampling_grid, angle_xy, angle_z = Sampling_grid(a, e, r, x, y, z, self.random_rotation[self.info[ID]['file']])
            sampling_grid_original = sampling_grid.copy()
            sampling_grid = sampling_grid+80
        
        
            'sample image slice'
            xx = np.arange(160)
            yy = np.arange(160)
            zz = np.arange(160)
            interp_arr = interpn((xx, yy, zz), self.vol_store[self.info[ID]['file']], np.transpose(sampling_grid.reshape((3,160*160))), bounds_error=False, fill_value=0)
            img_slice = interp_arr.reshape((160,160))
            interp_arr_mask = interpn((xx, yy, zz), self.mask_store[self.info[ID]['file']], np.transpose(sampling_grid.reshape((3,160*160))), bounds_error=False, fill_value=0, method='nearest')
            mask = interp_arr_mask.reshape((160,160))
            mask_original = mask.copy()
            
            
            'random version of skull mask'
            contours = cv2.findContours(cv2.convertScaleAbs(mask).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(contours)>1:
                mask = convex_hull_image(mask).astype(int)
                contours = cv2.findContours(cv2.convertScaleAbs(mask).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(contours)>1:
                    print('more than 1 contour')
            contours = cv2.approxPolyDP(contours[0], 3, True)
            centers, radius = cv2.minEnclosingCircle(contours)
            circle = create_circular_mask(160, 160, center=centers, radius=random.uniform(radius,80))
            
            mode_num = random.uniform(0,3)
            if mode_num<1:
                img_slice =  img_slice*mask
            elif mode_num>=1 and mode_num<2:
                img_slice =  img_slice*circle
            
            
            'Scale augmentation (dont do this if you dont want the prediction to be scale invariant)' 
            mask = mask_original
            col_index = [n for n,i in enumerate(list(mask.sum(0))) if i>0 ]
            row_index = [n for n,i in enumerate(list(mask.sum(1))) if i>0 ]
            r1 = row_index[0]
            r2 = row_index[-1]
            c1 = col_index[0]
            c2 = col_index[-1]
    
            rescale_factor = random.uniform(1, min(160/max(r2-r1, c2-c1), 1.8))
            img_slice = resize(img_slice, (int(img_slice.shape[0]*rescale_factor), int(img_slice.shape[1]*rescale_factor))
                                , order=3, mode='constant')
            mask = resize(mask, (int(mask.shape[0]*rescale_factor), int(mask.shape[1]*rescale_factor))
                                , order=0, mode='constant')
            total_r = 160-int(r2*rescale_factor)+int(r1*rescale_factor)
            total_c = 160-int(c2*rescale_factor)+int(c1*rescale_factor)
            r1_lim = min(160-int(r2*rescale_factor)+int(r1*rescale_factor), int(r1*rescale_factor))
            c1_lim = min(160-int(c2*rescale_factor)+int(c1*rescale_factor), int(c1*rescale_factor))
            r2_lim = min(160-int(r2*rescale_factor)+int(r1*rescale_factor), img_slice.shape[0]-int(r2*rescale_factor))
            c2_lim = min(160-int(c2*rescale_factor)+int(c1*rescale_factor), img_slice.shape[1]-int(c2*rescale_factor))
            
            r_pad1 = random.randint(total_r-r2_lim, r1_lim)
            c_pad1 = random.randint(total_c-c2_lim, c1_lim)
            r_pad2 = 160-int(r2*rescale_factor)+int(r1*rescale_factor)-r_pad1
            c_pad2 = 160-int(c2*rescale_factor)+int(c1*rescale_factor)-c_pad1
            img_slice = img_slice[int(r1*rescale_factor)-r_pad1:int(r2*rescale_factor)+r_pad2, 
                                  int(c1*rescale_factor)-c_pad1:int(c2*rescale_factor)+c_pad2]
#            print(np.min(img_slice), np.max(img_slice))
            
            
            'Translation augmentation (dont do this if you dont want the prediction to be in-plane translation invariant)'
            col_index = [n for n,i in enumerate(list(mask.sum(0))) if i>0 ]
            row_index = [n for n,i in enumerate(list(mask.sum(1))) if i>0 ]
            r1 = row_index[0]
            r2 = row_index[-1]
            c1 = col_index[0]
            c2 = col_index[-1]
            row_shift = random.uniform(-r1,160-r2)
            col_shift = random.uniform(-c1,160-c2)
            img_slice = ndimage.shift(img_slice, (row_shift, col_shift))
            img_slice[img_slice<0]=0
            
            
            'Intensity augmentation'
            img_slice = img_slice*uniform(0.9,1.1)*np.random.uniform(low=0.99, high=1.01, size=img_slice.shape)
            augmentation = iaa.Sequential([
                    iaa.GammaContrast((0.5, 2.0)),
#                    iaa.CropAndPad(px=(-10,30), sample_independently=False),
#                    iaa.PiecewiseAffine(scale=(0, 0.01)),
                    ])
            augmentation = iaa.GammaContrast((0.5, 2.0))
            try:
                img_slice = augmentation.augment_image(np.expand_dims(img_slice,axis=-1))
            except:
                print(np.min(img_slice), np.max(img_slice))
                img_slice = augmentation.augment_image(np.expand_dims(img_slice,axis=-1))
            img_slice = np.squeeze(img_slice)

            'Final preprocessing'
            img_slice = image_normalize(img_slice)
            img_slice = np.expand_dims(img_slice, axis=0)
            images[count,:,:,:] = img_slice[np.newaxis,::]
            
            'Get groundtruth'
            sampling_grid = sampling_grid-80
            
            y1[count,:] = np.array([sampling_grid_original[0,80,80],
                                   sampling_grid_original[1,80,80],
                                   sampling_grid_original[2,80,80]])
            y2[count,:] = np.array([sampling_grid_original[0,159,0],
                                   sampling_grid_original[1,159,0],
                                   sampling_grid_original[2,159,0]])
            y3[count,:] = np.array([sampling_grid_original[0,159,159],
                                   sampling_grid_original[1,159,159],
                                   sampling_grid_original[2,159,159]])
            
        
        if self.mode=='training' or self.mode=='validation':
            return images, y1, y2, y3
        else:
            return images, y1, y2, y3, img_vol, self.info[ID]['file']


    def create_list_and_ID(self, images):
        r_z = np.linspace(0,270, num=4)     #4 random in-plane rotation
        t_z = np.linspace(0,1, num=self.dist_num)
        azimuth = []
        elevation = []
        for i in range(-self.sample_num//2, self.sample_num//2, 1):
            a1 = 2*math.pi*i/((np.sqrt(5)+1)/2)
            a2 = (2*i)/(2*self.sample_num/2+1)
            a2 = np.arcsin(a2)
            azimuth.append(a1)
            elevation.append(a2)

        plane_id = 0
        list_IDs = []
        info = {}
        self.random_rotation = {}
        self.dist_store={}      #store the maximum distance along each normal
        self.vol_store = {}
        self.mask_store = {}
        combined = [r_z, t_z, azimuth]
        combined = list(itertools.product(*combined))
        combined = [list(i) for i in combined]
        for plane in combined:
            plane.append(elevation[azimuth.index(plane[-1])])
        self.sampling_params = combined
        numbers = list(np.arange(len(combined)))
        print(self.mode, end=' ')
        sys.stdout.flush()
        for n, fname in enumerate(images):
            mat = sio.loadmat(fname) # mat contains the img volume and the binar mask of the skull (region of interest)
            img_vol = np.squeeze(mat['img_brain'])
            img_mask = np.squeeze(mat['img_brain_mask'])
            img_mask[img_mask>0]=1
            self.vol_store.update({fname:img_vol})
            self.mask_store.update({fname:flood_fill_hull(img_mask)})
            
            'random rotation'
            pts=50000000000
            i=random.randint(-pts, pts)
            a1 = 2*math.pi*i/((np.sqrt(5)+1)/2)
            a2 = (2*i)/(2*pts+1)
            a2 = np.arcsin(a2)
            norm_vector = find_norm_vector(a1, a2)
            norm_vector = norm_vector/np.linalg.norm(norm_vector)
            rotation_random = R_2vect(np.array([0,0,1]), norm_vector)
            self.random_rotation.update({fname:rotation_random.copy()})
            
            'find maximum translation along each normal'
            temp_dict = find_dist(np.squeeze(mat['img_brain_mask'].astype(int)), azimuth, elevation, rotation_random)
            self.dist_store.update({fname:temp_dict})
            print(n, end=' ')
            sys.stdout.flush()
#            for key in temp_dict.keys():
#                print(temp_dict[key][0], -temp_dict[key][1], end="")
            
            random.shuffle(numbers)
            l = [numbers[i:i + self.batch_size] for i in range(0, len(numbers), self.batch_size) ]
            
            for param_ids in l:
                temp = {}
                list_IDs.append(plane_id)
                temp.update({'file':fname})
                temp.update({'param_ids':param_ids})
                info.update({plane_id:temp})
                plane_id+=1
        print('')
        
        random.shuffle(list_IDs)
        self.list_IDs = list_IDs
        self.info = info
        self.shuffle_list()
        
    def shuffle_list(self):
        random.shuffle(self.list_IDs)
        random.shuffle(self.sampling_params)
        
