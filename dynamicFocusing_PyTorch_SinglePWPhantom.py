"""
2020-03-09
Porting to PyTorch the official Tensorflow implementation of the dynamic focusing layer
for US RX beamforming proposed in (direct translation):

"Learning beamforming in ultrasound imaging", Proc. MIDL 2019.

Some of the code is based on the official implementation
of the following paper:
Jaderberg et al., Spatial Transformer Networks, NIPS 2015.
"""

# PyTorch DNN imports
import torch
import torch.nn
import torch.optim

# Numpy/Scipy and other imports
from scipy.io import loadmat
import scipy.interpolate
from scipy.signal import  convolve2d
import numpy as np
import random
import os

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# -------------------------------------------------------------------------------------------------
## Seed all the random number generators
def seed_prng(prng_seed, device):
    # Do the below to ensure reproduceability - from the last comment at
    # https://discuss.pytorch.org/t/random-seed-initialization/7854/18
    # NOTE: I didn't do the num_workers thing they suggested, but reproducibility
    # was obtained without it
    # NOTE: According to FAIR fastMRI code, the three below lines will suffice for reproducibility... it doesn't set the CUDA seeds.
    np.random.seed(prng_seed)
    random.seed(prng_seed)
    torch.manual_seed(prng_seed)

    # if you are using GPU # This might not be necessary, as per https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
    if 'cuda' in device.type:
        torch.cuda.manual_seed(prng_seed)
        torch.cuda.manual_seed_all(prng_seed) 
        torch.backends.cudnn.enabled = True # This was originally false. Changing it to true still seems to work.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 

# -------------------------------------------------------------------------------------------------
## Function to plot the Fourier content of a given slice
def plot_freq_content(data_slice, fs, fig_title, subfig_idx, color, axes):
    # axes[0, subfig_idx].set_title(fig_title)
    # axes[0, subfig_idx].magnitude_spectrum(data_slice, Fs=fs, color=color)
    axes[subfig_idx].set_title(fig_title)
    axes[subfig_idx].magnitude_spectrum(data_slice, Fs=fs, color=color)
    axes[subfig_idx].set_xlim(-12e6, 12e6)

# -------------------------------------------------------------------------------------------------
## Function to load and process the requested MAT file
def load_data():
    data_dir = 'dynamicFocusing_PyTorch_SinglePWPhantom'
    file_name = '20190812-181930/42_layer0_idx42_BDATA_RF.mat'
    seq_file_name = '20190812-181930/Sequence.mat'
    mat_data = loadmat(os.path.join(data_dir, file_name))
    seq_data = loadmat(os.path.join(data_dir, seq_file_name))

    labels_dict = {}
    # curr_label = mat_data['curr_label'][0][0]
    labels_dict['c'] = seq_data['Parameter'][0]['speedOfSoundMps'].item().item() # Want to remove all the extraneous array([]) objects and retrieve only the value out for sanity
    # labels_dict['r'] = curr_label['r'].item().item() 
    # labels_dict['z'] = curr_label['z'].item().item() 
    # labels_dict['x'] = curr_label['x'].item().item() 
    
    probe_dict = {}
    probe_dict['f0']         = seq_data['Tw'][0]['freqMHz'].item().item() * 1e6
    probe_dict['fs']         = seq_data['System'][0]['Parameters']['sampleFreqMHz'].item().item() * 1e6
    probe_dict['dt']         = 1/probe_dict['fs']
    probe_dict['N_elements'] = seq_data['System'][0]['Transducer']['elementCnt'].item().item()
    probe_dict['pitch']      = seq_data['System'][0]['Transducer']['elementPitchCm'].item().item()/100
    probe_dict['elemCoor'] = np.arange(probe_dict['N_elements'])*probe_dict['pitch']
    probe_dict['elemCoor'] = probe_dict['elemCoor'] - (probe_dict['elemCoor'][0]+probe_dict['elemCoor'][-1])/2
    # Get the lowpass filter weights and lag estimate from MATLAB - you calculate those temrs in matlab, and just read it from a mat file here - from the simulated data
    lpf_and_lag = loadmat(os.path.join(data_dir, 'matlab_deets.mat'))
    probe_dict['lpf_weights'] = np.squeeze(lpf_and_lag['filter_weights'])
    # probe_dict['lag'] = lpf_and_lag['lag'].item() # Didn't use a lag for phantom data
    probe_dict['lag'] = 0 # Didn't use a lag for phantom data

    data_dict = {}
    # Attempt 1: Predictable loading
    data_dict['v_2']        = np.reshape(np.swapaxes(mat_data['AdcData_frame000'],0,1), (mat_data['AdcData_frame000'].shape[1], mat_data['AdcData_frame000'].shape[0] * mat_data['AdcData_frame000'].shape[2]), order='F').astype('float32')
    # # Attempt 2: Rearrange in MATLAB then process it here
    # alp_data = loadmat(os.path.join(data_dir, 'alp_data.mat'))
    # data_dict['v_2']        = alp_data['v_2']    

    # For the below, could set it to 0 or use t_in=time-q/(channel_data.sound_speed)-System.Transducer.delayOffsetUsec*10^-6; as in ustb/@alpinion/read_CPWC.m
    # Doesn't make much of a difference as far as I can see
    # # data_dict['t_2']        = mat_data['t_2'].item() # For simulatiions
    # data_dict['t_2']        = 0 # Initial time is 0 for this
    data_dict['t_2']        = -seq_data['System'][0]['Transducer']['delayOffsetUsec'].item().item()*1e-6
    data_dict['min_sample'] = data_dict['t_2']*probe_dict['fs']
    data_dict['t_in']       = (np.arange(data_dict['v_2'].shape[0]) + data_dict['min_sample'])/probe_dict['fs'] - probe_dict['lag']*probe_dict['dt']
    data_dict['min_sample_wlag'] = data_dict['t_2'] - probe_dict['lag']*probe_dict['dt']
    data_dict['z_in']       = data_dict['t_in']*labels_dict['c']/2

    ## Step 1: Downmixed data 
    # Multiply raw RF data with a complex exponential to bring it down in the frequency domain
    data_dict['complex_sinusoid'] = np.exp(-1j * 2 * np.pi * probe_dict['f0'] * data_dict['t_in'])
    data_dict['v_2_downmixed'] = data_dict['v_2'] * np.tile(np.reshape(data_dict['complex_sinusoid'], (len(data_dict['complex_sinusoid']), 1)), (1, probe_dict['N_elements']))
    ## Step 2: Low pass filter the downmixed data 
    data_dict['v_2_downmixed_lpf'] = convolve2d(data_dict['v_2_downmixed'], np.reshape(probe_dict['lpf_weights'], (len(probe_dict['lpf_weights']), 1)), mode='same')
    ## Step 3: Downsample?? TODO - also implement its plotting below
    data_dict['v_2_downmixed_lpf_resampled'] = data_dict['v_2_downmixed_lpf']
    ## Step 4: Normalization - added here in the Python script - was not there in the MATLAB script
    data_dict['v_2_downmixed_lpf_resampled'] = data_dict['v_2_downmixed_lpf_resampled']/np.absolute(data_dict['v_2_downmixed_lpf_resampled']).max()
    
    to_plot = 0
    if to_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        plot_freq_content(data_dict['complex_sinusoid'], probe_dict['fs'], 'Complex Sinusoid', (0,0), 'C0', axes)
        plot_freq_content(np.squeeze(data_dict['v_2'][:,64]), probe_dict['fs'], 'Original Data', (0,1), 'C1', axes)
        plot_freq_content(np.squeeze(data_dict['v_2_downmixed'][:,64]), probe_dict['fs'], 'Downmixed Data', (1,0), 'C2', axes)
        plot_freq_content(np.squeeze(data_dict['v_2_downmixed_lpf'][:,64]), probe_dict['fs'], 'Downmixed+LPF Data', (1,1), 'C3', axes)

    return data_dict, probe_dict, labels_dict

# Implementation of gather_nd from tensorflow implemented in PyTorch
# Tried using the below - was too limited for my purposes:
# https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502/7
# Just impelemnted it myself
def gather_nd(params, indices):
    """
    Utility function to implement tensorflow's
    gather_nd function's action in pytorch 
    Input
    -----
    - params:  Tensor of K-1 dimensions - dim_{0}, ..., dim_{K-2}
               This would contain the pixel values you are sampling from
    - indices: Tensor of K dimensions  - dim_{0}, ..., dim_{K-1} 
               NOTE: Length along the K-1th dimension) == K-1 HAS TO BE TRUE
               (i.e., indices[i_{1},...,i_{K-2}] as a K-1 length vector containing the 
               coordinates of the element from params to place into img_mod)
    Returns
    -------
    - img_mod: Tensor of same shape as params, sampled from elements of params
    """
    # Assert that (length along the K-1th dimension) == K-1
    assert len(params.shape) == indices.shape[-1]

    # Store a vectrorized version of each dimension's coordinate in a list
    dim_list = []
    for idx in range(indices.shape[-1]):
        dim_list.append(indices[:,:,:,:,:,idx].view(-1).long())
    
    # Take the appropriate elements out of params using indices and put it into img_mod
    img_mod = params[dim_list].view(params.shape)
    return img_mod

def dynamic_focusing_layer(img_tensor, theta, data_dict, probe_dict, labels_dict, elemCoor, device, trainable=False):

    dims = img_tensor.shape # function to get the shape - returns [2, 8002, 128, 1, 1]
    theta = torch.from_numpy(theta).requires_grad_(requires_grad=trainable).to(device) # theta is a variable, initialized at theta_init
    # THIS IS WHAT IS TRAINABLE, NOT c...
    c = torch.from_numpy(np.squeeze(labels_dict['c']).astype(np.float32)).requires_grad_(requires_grad=False).to(device)  # 1540 m/s
    fs = torch.from_numpy(np.squeeze(probe_dict['fs']).astype(np.float32)).requires_grad_(requires_grad=False).to(device) # 3571428.5000 Hz


    # t = (torch.from_numpy(np.arange(dims[1], dtype=np.float32))/fs).requires_grad_(requires_grad=False).to(device)
    t = (torch.from_numpy( data_dict['t_in'].astype(np.float32) )).requires_grad_(requires_grad=False).to(device)
    # dims[1] is 652, the number of depth samples. np.arange(652) generates a (652,) array with entries [0,...,651]

    # w0 = torch.from_numpy(np.squeeze(2.0 * np.pi * specs['DemodulationFrequency']).astype(np.float32)).requires_grad_(requires_grad=False).to(device)
    w0 = torch.from_numpy(np.squeeze(2.0 * np.pi * probe_dict['f0']).astype(np.float32)).requires_grad_(requires_grad=False).to(device) # TODO: Not sold on this? Does bringing it to baseband break things?
    # specs['DemodulationFrequency'] is 2109527.587891

    ee, tt, ss = torch.meshgrid([elemCoor, t, elemCoor]) # TODO: Check requires_grad properties here...
    ee = torch.transpose(ee,0,1)
    tt = torch.transpose(tt,0,1)
    ss = torch.transpose(ss,0,1)
    # Does not behave like MATLAB's meshgrid by default; need to swap the first two axes

    r = 0.5 * tt * c # Multiply each element of tt (which time the signal was sampled at) by c and divide by 2 to account for tx and rx
    # Yields the distance of each sample point from the

    # x_rx = r * torch.sin(ss) # elementwise multiply each pixel's distance by the sin(angle) it forms at the center of the array to get its x coordinate
    # z_rx = r * torch.cos(ss) # elementwise multiply each pixel's distance by the cos(angle) it forms at the center of the array to get its z coordinate
    x_rx = ss # TODO: Single plane wave only currently - later include dependence on theta
    z_rx = r

    delays_grid_t = (r+(torch.sqrt((x_rx-ee)**2+(z_rx)**2)))/c
    delays_grid_t = delays_grid_t - torch.from_numpy(np.array(data_dict['min_sample_wlag']).astype(np.float32)).to(device)
    # Might seem slightly more complicated, but is simple really:
    # 1) For each sample point, you need the distance of it from each probe element in the x axis - this is (x_rx-ee)
    # 2) For each sample point, you then need the distance of it from each probe element in the z axis - this is z_rx
    # 3) Simply square each coordinate, add, take square root - Euclidean distance
    # 4) TODO: For some reason, you need to add r here... figure it out why... Derive it step by step once! Probably because it's the transmit time?
    # I think the second (long) term of the above distance is the distance between a point and each element of the probe on receive... After the point scatters, this is what is required for the additional delays!
    # I think the first (r) term of the above distance is simply the distance between each point and center of the probe - this is what matters on transmit...
    # 5) Once you have total distance, simply divide by c to get the total delay (time) that, for the total computed distance, it takes that needs to be compensated
    delays_grid = delays_grid_t*fs # Multiply delay (s) by fs (sampling rate, Hz) to get delay in **samples**
    # delays_grid = torch.clamp(delays_grid,min=0.0, max=dims[1]-1.0)
    delays_grid = torch.clamp(delays_grid,min=0.0, max=dims[1]-2.0) # NOTE: Original code was till dims[1]-1.0. Had to change it to 2.0 here with my implemetation of gather_nd.
    # Given a tensor delays_grid, this ensures that if the values in it are outside the interval [0, 652-1], they are clipped to 0 or 651

    # sample input with grid to get output
    out_img_tensor = bilinear_sampler(img_tensor, delays_grid) # Reminder: img_tensor is shape=(2, 652, 64, 140, 1), delays_grid is (652, 64, 140)
    # These capture the additional phase shift introduced
    cos_phi = torch.cos(w0*(delays_grid_t-tt)).unsqueeze(-1)
    sin_phi = torch.sin(w0*(delays_grid_t-tt)).unsqueeze(-1)
    # Both are sized as (652, 64, 140, 1)

    real,imag = torch.split(out_img_tensor, split_size_or_sections=1, dim=0) # Split the two channels on the first axis into two
    IQx = real*cos_phi-imag*sin_phi
    IQy = real*sin_phi+imag*cos_phi
    out = torch.cat([IQx,IQy],axis=0)
    # out = out_img_tensor
    return out
 
# Copied from https://github.com/kevinzakka/spatial-transformer-network
def get_pixel_value(img, h, e, s):
    """
    Original - Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Modified - Utility function to get pixel value for coordinate
    vectors h, w, and d from a  5D tensor image.    
    Input
    -----
    - img: tensor of shape (B, H, W, C) (Original)
    - img: tensor of shape (B, H, E, S, C) (Modified)
    - x:      flattened tensor of shape (B*H*W,) (Original)
    - y:      flattened tensor of shape (B*H*W,) (Original)
    - h:      tensor of shape (1, H, E, S) (Modified)
    - rx_ele: tensor of shape (1, H, E, S) (Modified)
    - sc:     tensor of shape (1, H, E, S) (Modified)
    Returns
    -------
    - output: tensor of shape (B, H, W, C) (Original)
    - output: tensor of shape (B, H, E, S, C) (Modified)
    """
    # img is (2, 652, 64, 140, 1), h & e & s are all (1, 652, 64, 140), 
    shape           = img.shape 
    batch_size      = shape[0] # (B)
    height          = shape[1] # (H)
    num_rx_elements = shape[2] # (E)
    scanlines       = shape[3] # (S)
    channels        = shape[4] # (C)

    h = h.unsqueeze(4) # Add the tail dimension previously lacking from h,w,d here
    e = e.unsqueeze(4)
    s = s.unsqueeze(4)
    
    batch_idx = torch.arange(0, batch_size).to(img.device)
    batch_idx = batch_idx.view([batch_size, 1, 1, 1, 1]) # So (2,1,1,1,1)
    b = batch_idx.expand([-1, height, num_rx_elements, scanlines, channels]).type(torch.int32) # And tile that too...

    h = h.expand([batch_size, -1, -1, -1, -1]) # Tile it again for h,w,d
    e = e.expand([batch_size, -1, -1, -1, -1])
    s = s.expand([batch_size, -1, -1, -1, -1])
    
    indices = torch.stack([b, h, e, s, torch.zeros(b.shape, dtype=torch.int32).to(img.device)], dim=-1).type(torch.int32) # Now this monstrosity is (2, 652, 64, 140,1,5)
    # torch.stack - Concatenates sequence of tensors along a new dimension.
    # torch.cat - Concatenates the given sequence of seq tensors in the given dimension.
    # So if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4) 
    # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
    return gather_nd(img, indices) # Tensorflow operation implemented in PyTorch
    # Basically, my understanding is that this, based on the indices, 
    # fills an img sized object with the right samples i.e. looks at each entry of img, picks the right
    # sample to put there, and places it there


# Copied from https://github.com/kevinzakka/spatial-transformer-network
def bilinear_sampler(img, delays_grid):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout. (Original)
    - img: batch of images in (B, H, E, S, C) layout. (Modified)
    - grid: x, y which is the output of affine_grid_generator. (Original)    
    - delays_grid: tensor which specifies required delays and is of size (H,E,S). (Modified)
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid. (Original)
    - out: interpolated images according to delays_grid. Same size as grid. (Modified)
    """
    shape = img.shape
    H = shape[1] # =652, the number of pixels in the depth dimension
    E = shape[2] # =64, the number of probe elements
    S = shape[3] # =140, the number of thetas
    # max_d = tf.cast(D, 'int32') # typecasts as int32
    # max_w = tf.cast(W, 'int32') # typecasts as int32
    max_s = S # 
    max_e = E # 
    s = torch.arange(0, max_s).to(img.device) # like np.arange -  [0, ..., 139=max_d-1]
    e = torch.arange(0, max_e).to(img.device) # like np.arange -  [0, ..., 63]

    # grab 4 nearest points to delays points
    e = e.view([1,E,1]) # input is of dimension = (64), output is of dimension (1,64,1)
    e = e.expand([H,-1,S]) # And now it is tiled to be (652,64,140)
    # NOTE: 1) expand returns a new view of the self tensor with singleton dimensions expanded to a larger size. (as opposed to repeat() which creates a copy)
    #       2) Passing -1 as the size for a dimension means not changing the size of that dimension.
    e = e.unsqueeze(0).type(torch.int32) # And now it is (1,652,64,140)
    h0 = torch.floor(delays_grid).type(torch.int32)
    # 1) Take the floor to discretize the delays_grid - so now the sample corresponding to each element of 
    # the delays_grid tensor is a discrete integer sample number
    # 2) Typecast to int32
    h0 = h0.unsqueeze(0) # it's now (1,652,64,140)
    h1 = h0 + 1 # adding 1 to it for some reason... AHA! Since we are doing bilinear interpolation, need the next closest one as well...

    s = s.view([1,1,S]) # Do similar stuff to d as well
    s = s.expand([H,E,-1])
    s = s.unsqueeze(0).type(torch.int32)

    # get pixel value at NN coords
    Ia = get_pixel_value(img, h0, e, s) # See the function for a more intricate explanation...
    Ib = get_pixel_value(img, h1, e, s)
    # both the above are sized as (2,652,64,140,1)

    # recast as float for delta calculation
    h0 = h0.type(torch.float32) # Make it back into float now
    h1 = h1.type(torch.float32)

    # calculate deltas
    wa = h1 - delays_grid # This is the bilinear sampling step - gets your weights - it's going to be fractional
    wb = delays_grid - h0
    # both the above are sized as (1,652,64,140)

    # add dimension for addition
    wa = wa.unsqueeze(-1)
    wb = wb.unsqueeze(-1)
    # so need to resize it to (1,652,64,140,1)

    # compute output
    out = wa * Ia + wb * Ib # elementwise multiplication

    return out

if __name__=='__main__':

    # Set if the layer is trainable or not
    trainable = False

    # Load the data to beamform
    data_dict, probe_dict, labels_dict = load_data()

    # test script for the function
    dims = [data_dict['v_2_downmixed_lpf_resampled'].shape[0], data_dict['v_2_downmixed_lpf_resampled'].shape[1], data_dict['v_2_downmixed_lpf_resampled'].shape[1]]
    # 652 is the number of depth pixels
    # 64 is the number of probe elements - so for each depth for every scanline gets 64 observed values that are summed over
    # 140 is the number of scanlines - this is the number of width pixels. In this case, MLAs are used - so think of it more 
    # as transmit events, though they seem to actually just be using SLAs (i.e.) just regular phased array imaging.

    # load the IQ raw data
    I = np.real(data_dict['v_2_downmixed_lpf_resampled'])
    I = np.array(I,dtype=np.float32) # Typecast it as a float32 array
    # I = np.expand_dims(np.expand_dims(I,0),4) # Change I's dimensions from a [652,64,140] array to a [1,652,64,140,1] array
    I = np.expand_dims(I,0) # Change I's dimensions from a [8002, 128] array to a [1, 8002, 128] array

    Q = np.imag(data_dict['v_2_downmixed_lpf_resampled'])
    Q = np.array(Q, dtype=np.float32) # Typecast it as a float32 array
    # Q = np.expand_dims(np.expand_dims(Q, 0), 4) # Repeat the same process with the Q channel
    Q = np.expand_dims(Q,0) # Change Q's dimensions from a [8002, 128] array to a [1, 8002, 128] array
    img = np.concatenate((I, Q), axis=0) # Concatenate the I and Q channels on the first axis to yield a [2,8002,128] array

    # Specify the device to run the code on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed the pseudo-random number generator
    prng_seed = 1337
    seed_prng(prng_seed=prng_seed, device=device)

    img_tensor = torch.from_numpy(np.repeat(img[:, :, :, np.newaxis, np.newaxis], img.shape[2], axis=3)).to(device) # Still shares memory with img - modifying it will modify img
    # img_tensor = img_tensor.to(device) # No longer shares memory with img - modifying it will not modify img

    # specs = loadmat('ph_specs.mat')
    # # specs is a dict with the following useful key-value pairs
    # # 1) specs - this isn't quite a dict nor an array, **BUT CAN BE ADDRESSED AS A DICT** and stores
    # #    a) NumProbeElements = 64
    # #    b) InputDataPitch = 680
    # #    c) NumOutputSamples = 652
    # #    d) nMLAs = 1
    # #    e) DemodulationFrequency = 2109527.587891
    # #    f) IQSampleRate = 3571428.571429
    # #    g) StartDepth = 0
    # #    h) SpeedOfSound = 1540
    # #    i) PitchOfProbeElement = 0.0003
    # # 2) thetaRX - is a [1,140] array storing the individual scanline transmit angles in radians - from -37.63537904001025 degrees to 37.63537904001025 degrees
    # theta = np.squeeze(specs['thetaRX']).astype(np.float32) # Extract specs['thetaRx'] and typecast it
    theta = np.array(0).astype(np.float32)

    # elemCoor = loadmat('3Sc_elem_pos.mat') # elemCoor['element_positions'] is the only interesting key value pair.
    # # It is a [64,3] array storing the coordinates of the elements of the probe. Second and third column are all zero.
    # # First column varies from -0.009449999999999998 to 0.009449999999999998 - consecutive elements are separated by PitchOfProbeElement defined above
    # # elemCoor = torch.tensor(np.array(elemCoor['elements_positions'][:, 0]).astype(np.float32), requires_grad=trainable) # Second and third column are irrelevant - they're tossed.
    # elemCoor = torch.from_numpy(np.squeeze(elemCoor['elements_positions'][:, 0]).astype(np.float32)).requires_grad_(requires_grad=trainable).to(device) # Second and third column are irrelevant - they're tossed.    
    # NOTE: torch.tensor() always copies data - many alternatives to prevent a copy
    elemCoor = torch.from_numpy(probe_dict['elemCoor'].astype(np.float32)).requires_grad_(requires_grad=trainable).to(device) 

    # specs = specs['specs']

    # get BFed data
    BFfmap = dynamic_focusing_layer(img_tensor, theta, data_dict, probe_dict, labels_dict, elemCoor, device, trainable=trainable)
    # It doesn't adjust speed of sound, rather adjusting angles of the individual scanlines i.e. theta variable

    # # test gradients with a dummy loss
    # loss = tf.reduce_mean(BFfmap - 5.0) # reduce_mean basically just takes the mean across all elements of a tensor
    # trainer = tf.train.AdamOptimizer(0.1)
    # opt = trainer.minimize(loss)
    # sess = tf.Session()
    # # sess = tf.compat.v1.Session()
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    # IQ = sess.run(BFfmap, feed_dict={input_fmap:img})
    # sess.run(opt, feed_dict={input_fmap: img})
    
    to_plot = 1
    if to_plot:
        # %matplotlib inline
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # Separate I and Q components
        IQ = BFfmap.detach().to("cpu").numpy()
        I_channel = IQ[0,:,:,:,0]
        Q_channel = IQ[1,:,:,:,0]
        # Generate complex I-Q data
        IQ_complex = I_channel + 1j*Q_channel
        # Sum across the transmits
        IQ_complex_sum = np.sum(IQ_complex, axis=1)
        IQ_complex_sum = IQ_complex_sum/np.max(abs(IQ_complex_sum))
        IQ_complex_sum_abs_logCompressed =  20*np.log10(abs(IQ_complex_sum))
        # imgplot = plt.imshow(IQ_complex_sum_abs_logCompressed, cmap='Greys', extent = [0, 30, 0, 20], aspect = 1.2)
        # imgplot = plt.imshow(-IQ_complex_sum_abs_logCompressed, cmap='Greys', extent = [probe_dict['elemCoor'].min(), probe_dict['elemCoor'].max(), data_dict['z_in'].max(), data_dict['z_in'].min()], aspect = 'equal', vmin=0, vmax=60)
        imgplot = plt.imshow(IQ_complex_sum_abs_logCompressed, cmap='Greys_r', extent = [probe_dict['elemCoor'].min(), probe_dict['elemCoor'].max(), data_dict['z_in'].max(), data_dict['z_in'].min()], aspect = 'equal', vmin=-60, vmax=0)
        plt.show()