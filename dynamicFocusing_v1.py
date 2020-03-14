"""
The official Tensorflow implementation of the dynamic focusing layer
for US RX beamforming proposed in:

"Learning beamforming in ultrasound imaging", Proc. MIDL 2019.

Some of the code is based on the official implementation
of the following paper:
Jaderberg et al., Spatial Transformer Networks, NIPS 2015.

"""

# Set the tensorflow version we are to run the code for
tf_version = 'v1'
# tf_version = 'v2'

if tf_version == 'v1':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
elif tf_version == 'v2':
    import tensorflow as tf

from scipy.io import loadmat
import numpy as np
import scipy.interpolate
import os

if tf_version == 'v1':
    layers = tf.layers
elif tf_version == 'v2':    
    layers = tf.keras.layers.Layer
    
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def dyanamic_focusing_layer(input_fmap, Theta, specs, elemCoor, trainable=False):

    dims = input_fmap._shape_as_list() # TF function to get the shape as a list - returns [2, 652, 64, 140, 1]
    theta_init = tf.constant(Theta) # Theta is an np array; theta_init is a tf.constant
    theta = tf.Variable(initial_value=theta_init, expected_shape=Theta.shape[-1], trainable=trainable) # theta is a variable, initialized at theta_init
    # Theta.shape[-1] is 140 as Theta.shape is (1,140). Can be trainable! THIS IS WHAT IS TRAINABLE, NOT c...

    c = tf.constant(np.squeeze(specs['SpeedOfSound']).astype(np.float32))
    # specs['SpeedOfSound'] is shaped (1,1) and is array(array(1540)), squeezing it makes it () i.e. just a scalar, 
    # but still array(array(1540)) 
    # using tf.constant generates a shape=() array i.e. a scalar

    fs = tf.constant(np.array(specs['IQSampleRate']).astype(np.float32))
    # was curious why np.array is used instead of np.squeeze... the results are different. 
    # np.array doesdn't seem to change it, but np.squeeze removes extraneous [[]]
    # using tf.constant generates a shape=(1,1) array
    # However, maybe in effect they're equivalent? Testing that by running the below instead of the above 
    # They seem to be equivalent # TODO: Be more rigorous
    # fs = tf.constant(np.squeeze(specs['IQSampleRate']).astype(np.float32)) 
    

    t = tf.constant(np.arange(dims[1], dtype=np.float32)) / fs
    # dims[1] is 652, the number of depth samples. np.arange(652) generates a (652,) array with entries [0,...,651]
    # using tf.constant generates a shape=(1,652) array
    w0 = tf.constant(np.array(2.0 * np.pi * specs['DemodulationFrequency']).astype(np.float32))
    # specs['DemodulationFrequency'] is 2109527.587891
    # using tf.constant generates a shape=(1,1) array

    ee, tt, ll = tf.meshgrid(elemCoor,t,theta) # Behaves like MATLAB's meshgrid by default; can also swap the first two axes if required
    # The input to tf.meshgrid is N one d arrays - not specific about if it is (length,) or (length,1) or (1,length)

    r = 0.5 * tf.multiply(tt ,c) # Multiply each element of tt (which time the signal was sampled at) by c and divide by 2 to account for tx and rx
    # Yields the distance of each sample point from the
    
    x_rx = tf.multiply(r,tf.sin(ll)) # elementwise multiply each pixel's distance by the sin(angle) it forms at the center of the array to get its x coordinate
    z_rx = tf.multiply(r,tf.cos(ll)) # elementwise multiply each pixel's distance by the cos(angle) it forms at the center of the array to get its z coordinate
    
    delays_grid_t = (r+(tf.sqrt(tf.square(x_rx-ee)+tf.square(z_rx))))/c
    # Might seem slightly more complicated, but is simple really:
    # 1) For each sample point, you need the distance of it from each probe element in the x axis - this is (x_rx-ee)
    # 2) For each sample point, you then need the distance of it from each probe element in the z axis - this is z_rx
    # 3) Simply square each coordinate, add, take square root - Euclidean distance
    # 4) TODO: For some reason, you need to add r here... figure it out why... Derive it step by step once!
    # I think the second (long) term of the above distance is the distance between a point and each element of the probe on receive... After the point scatters, this is what is required for the additional delays!
    # I think the first (r) term of the above distance is simply the distance between each point and center of the probe - this is what matters on transmit...
    # 5) Once you have total distance, simply divide by c to get the total delay (time) that, for the total computed distance, it takes that needs to be compensated
    delays_grid = delays_grid_t*fs # Multiply delay (s) by fs (sampling rate, Hz) to get delay in **samples**
    delays_grid = tf.clip_by_value(delays_grid,clip_value_min=0.0, clip_value_max=dims[1]-1.0)
    # Given a tensor delays_grid, this ensures that if the values in it are outside the interval [0, 652-1], they are clipped to 0 or 651

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, delays_grid) # Reminder: input_fmap is shape=(2, 652, 64, 140, 1), delays_grid is (652, 64, 140)
    # These capture the additional phase shift introduced... TODO: Understand it better! The additional compensation required is fuzzy
    cos_phi = tf.expand_dims(tf.cos(w0*(delays_grid_t-tt)),axis=3) 
    sin_phi = tf.expand_dims(tf.sin(w0*(delays_grid_t-tt)),axis=3)
    # Both are sized as (652, 64, 140, 1)

    real,imag = tf.split(out_fmap,num_or_size_splits=2,axis=0) # Split the two channels on the first axis into two
    IQx = real*cos_phi-imag*sin_phi
    IQy = real*sin_phi+imag*cos_phi
    out = tf.concat([IQx,IQy],axis=0)
    return out


# NOTE: Think this code is from the spatial transformer networks paper TODO: Verify!
def get_pixel_value(img, h,w,d):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    # img is (2, 652, 64, 140, 1), h & w & d are all (1, 652, 64, 140), 
    shape = img._shape_as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]
    channels = shape[4]

    h = tf.expand_dims(h,4) # Add the tail dimension previously lacking from h,w,d here
    w = tf.expand_dims(w,4)
    d = tf.expand_dims(d,4)
    
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1, 1)) # So (2,1,1,1,1)
    b = tf.tile(batch_idx, (1, height, width, depth, channels)) # And tile that too...

    h = tf.tile(h,(batch_size, 1, 1, 1, 1)) # Tile it again for h,w,d
    w = tf.tile(w, (batch_size, 1, 1, 1, 1))
    d = tf.tile(d, (batch_size, 1, 1, 1, 1))
    indices = tf.stack([b, h, w, d, tf.zeros(shape=b._shape_as_list(),dtype=tf.int32)], 5) # Now this monstrosity is (2, 652, 64, 140,1,5) 
    return tf.gather_nd(img, indices)
    # tf.gather_nd - https://www.tensorflow.org/api_docs/python/tf/gather_nd - look at the examples
    # Basically, my understanding is that this, based on the indices, 
    # fills an img sized object with the right samples i.e. looks at each entry of img, picks the right
    # sample to put there, and places it there


# NOTE: Think this code is from the spatial transformer networks paper TODO: Verify!
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
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    shape = img._shape_as_list()
    H = shape[1] # =652, the number of pixels in the depth dimension
    W = shape[2] # =64, the number of probe elements
    D = shape[3] # =140, the number of thetas
    max_d = tf.cast(D, 'int32') # typecasts as int32
    max_w = tf.cast(W, 'int32') # typecasts as int32
    d = tf.range(0,max_d) # like np.arange -  [0, ..., 139=max_d-1]
    w = tf.range(0,max_w) # like np.arange -  [0, ..., 63]

    # grab 4 nearest points to delays points
    w = tf.reshape(w,[1,W,1]) # input is of dimension = (64), output is of dimension (1,64,1)
    w = tf.tile(w,[H,1,D]) # And now it is tiled to be (652,64,140)
    w = tf.expand_dims(w,0) # And now it is (1,652,64,140)
    h0 = tf.cast(tf.floor(delays_grid), 'int32')
    # 1) Take the floor to discretize the delays_grid - so now the sample corresponding to each element of 
    # the delays_grid tensor is a discrete integer sample number
    # 2) Typecast to int32
    h0 = tf.expand_dims(h0,0) # it's now (1,652,64,140)
    h1 = h0 + 1 # adding 1 to it for some reason... AHA! Since we are doing bilinear interpolation, need the next closest one as well...
    # Actually, doesn't make complete sense *shrugs*

    d = tf.reshape(d,[1,1,D]) # Do similar stuff to d as well
    d = tf.tile(d,[H,W,1])
    d = tf.expand_dims(d,0)

    # get pixel value at NN coords
    Ia = get_pixel_value(img,h0,w,d) # See the function for a more intricate explanation...
    Ib = get_pixel_value(img,h1,w, d)
    # both the above are sized as (2,652,64,140,1)

    # recast as float for delta calculation
    h0 = tf.cast(h0, 'float32') # Make it back into float now
    h1 = tf.cast(h1, 'float32')

    # calculate deltas
    wa = h1 - delays_grid # This is the bilinear sampling step - gets your weights - it's going to be fractional
    wb = delays_grid - h0
    # both the above are sized as (1,652,64,140)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=4) 
    wb = tf.expand_dims(wb, axis=4)
    # so need to resize it to (1,652,64,140,1)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib])
    # pretty sure the * operator is elementwise multiplication
    # TODO: Replace with with tf.multiply to verify

    return out


if __name__=='__main__':

    # test script for the function
    dims = [652,64,140] 
    # 652 is the number of depth pixels
    # 64 is the number of probe elements - so for each depth for every scanline gets 64 observed values that are summed over
    # 140 is the number of scanlines - this is the number of width pixels. In this case, MLAs are used - so think of it more 
    # as transmit events, though they seem to actually just be using SLAs (i.e.) just regular phased array imaging.

    # load the IQ raw data
    I = loadmat('./sample_data/I/sample1.mat') # Loads a dict, with the only important key-value pair being the I channel data
    I = np.array(I['I'],dtype=np.float32) # Extract the I channel data from the dictionary and typecast it as a float32 array
    I = np.expand_dims(np.expand_dims(I,0),4) # Change I's dimensions from a [652,64,140] array to a [1,652,64,140,1] array

    Q = loadmat('./sample_data/Q/sample1.mat')
    Q = np.array(Q['Q'], dtype=np.float32)
    Q = np.expand_dims(np.expand_dims(Q, 0), 4) # Repeat the same process with the Q channel
    img = np.concatenate((I, Q), axis=0) # Concatenate the I and Q channels on the first axis to yield a [2,652,64,140,1] array

    input_fmap = tf.placeholder(dtype=tf.float32, shape=[2, 652, 64, 140, 1]) 
    # Inserts a placeholder for a tensor that will be always fed i.e. need to always provide an actual variable of the correct 
    # size every time we operate on input_fmap

    specs = loadmat('ph_specs.mat')
    # specs is a dict with the following useful key-value pairs
    # 1) specs - this isn't quite a dict nor an array, **BUT CAN BE ADDRESSED AS A DICT** and stores
    #    a) NumProbeElements = 64
    #    b) InputDataPitch = 680
    #    c) NumOutputSamples = 652
    #    d) nMLAs = 1
    #    e) DemodulationFrequency = 2109527.587891
    #    f) IQSampleRate = 3571428.571429
    #    g) StartDepth = 0
    #    h) SpeedOfSound = 1540
    #    i) PitchOfProbeElement = 0.0003
    # 2) thetaRX - is a [1,140] array storing the individual scanline transmit angles in radians - from -37.63537904001025 degrees to 37.63537904001025 degrees
    theta = np.array(specs['thetaRX']).astype(np.float32) # Extract specs['thetaRx'] and typecast it

    elemCoor = loadmat('3Sc_elem_pos.mat') # elemCoor['element_positions'] is the only interesting key value pair. 
    # It is a [64,3] array storing the coordinates of the elements of the probe. Second and third column are all zero.
    # First column varies from -0.009449999999999998 to 0.009449999999999998 - consecutive elements are separated by PitchOfProbeElement defined above
    elemCoor = tf.constant(np.array(elemCoor['elements_positions'][:, 0]).astype(np.float32)) # Second and third column are irrelevant - they're tossed.
    # Just keep the [64,] length array corresponding to the first column, typecase it as float32 and use tf.constant

    specs = specs['specs']

    # get BFed data
    BFfmap = dyanamic_focusing_layer(input_fmap, theta, specs, elemCoor, trainable=True) # TODO: Further investigate trainable=True here
    # It doesn't adjust speed of sound, rather adjusting angles of the individual scanlines i.e. theta variable

    # test gradients with a dummy loss
    loss = tf.reduce_mean(BFfmap - 5.0) # reduce_mean basically just takes the mean across all elements of a tensor
    trainer = tf.train.AdamOptimizer(0.1)
    opt = trainer.minimize(loss)
    sess = tf.Session()
#     sess = tf.compat.v1.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    IQ = sess.run(BFfmap, feed_dict={input_fmap:img})
    sess.run(opt, feed_dict={input_fmap: img})
    # %matplotlib inline
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # Separate I and Q components
    I_channel = IQ[0,:,:,:,0]
    Q_channel = IQ[1,:,:,:,0]
    # Generate complex I-Q data
    IQ_complex = I_channel + 1j*Q_channel
    # Sum across the transmits
    IQ_complex_sum = np.sum(IQ_complex, axis=1)
    IQ_complex_sum = IQ_complex_sum/np.max(abs(IQ_complex_sum))
    IQ_complex_sum_abs_logCompressed =  20*np.log10(abs(IQ_complex_sum))
    imgplot = plt.imshow(IQ_complex_sum_abs_logCompressed, cmap='Greys', extent = [0, 30, 0, 20], aspect = 1.2)
    plt.show()
    # Scan Convert it using scipy.interpolate.griddata
    input_img = IQ_complex_sum # (652, 140)

    #theta already defined above
    c = np.squeeze(specs['SpeedOfSound'])
    fs = np.array(specs['IQSampleRate'])    
    t = (np.arange(input_img.shape[0], dtype=np.float32)) / fs # .shape[0] is the depth dimension = 652
    w0 = (np.array(2.0 * np.pi * specs['DemodulationFrequency']).astype(np.float32))

    ll, tt = np.meshgrid(theta, t) 
    r = 0.5 * np.multiply(tt ,c) 
   
    x_rx_input = r * np.sin(ll) # elementwise multiply each pixel's distance by the sin(angle) it forms at the center of the array to get its x coordinate
    z_rx_input = r * np.cos(ll) # elementwise multiply each pixel's distance by the cos(angle) it forms at the center of the array to get its z coordinate

    x_rx_input_vectorized = x_rx_input.reshape(np.prod(x_rx_input.shape[:]))
    z_rx_input_vectorized = z_rx_input.reshape(np.prod(z_rx_input.shape[:]))

    x_rx_output, z_rx_output = np.meshgrid(np.linspace(x_rx_input.min(), x_rx_input.max(), 512), np.linspace(z_rx_input.min(), z_rx_input.max(), 512))
    output_img = scipy.interpolate.griddata((x_rx_input_vectorized, z_rx_input_vectorized), input_img.reshape(np.prod(input_img.shape[:])), (x_rx_output, z_rx_output), method = 'linear', fill_value=0)
    IQ_complex_sum = output_img/np.max(abs(output_img))
    IQ_complex_sum_abs_logCompressed =  20*np.log10(abs(IQ_complex_sum))
    imgplot = plt.imshow(IQ_complex_sum_abs_logCompressed, cmap='Greys', extent = [x_rx_output.min(), x_rx_output.max(), z_rx_output.max(), z_rx_output.min()], aspect = 'equal')
    plt.show()
