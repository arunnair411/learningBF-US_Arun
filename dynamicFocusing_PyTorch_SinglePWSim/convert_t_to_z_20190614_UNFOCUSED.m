% % Author: Arun Asokan Nair
% % Date created: 201a7-10-22
% % Date modified: 2019-06-22
% % Purpose: This script reformats the received channel data to create a
% % datagrid in terms of z coordinates by mapping received times through the
% % speed of sound. 
% % Note: The beamformed image from raw rf channel data matches exactly
% with the simulation parameters - cyst is of the right radius, and at the
% right location
%% Clear workspace, close plots, clear command history, add requisite paths
close all; clearvars; clc;
addpath('utils/');
addpath('/data/manish/Programs/ustb/');            % USTB

%% Things to decide before running this script
% #0 - Specify if the code is being run on peterchin.jhu.edu or
% pulsegpu.hwcampus.jhu.edu
% data_prefix = '../../Dataset_creation/'; % Use this if it is being run on peterchin.jhu.edu
data_prefix = '../Dataset_creation/'; % Use this if it is being run on pulsegpu.hwcampus.jhu.edu
% #1 - Specify params, labels files to inform the processing and
% channel_data directory containing the raw data. Also transmit frequency 
% and sampling frequency of the probe

% % % Case 0 - Deterministic grid - 4Mhz, with 100Mhz sampling frequency
% % The below loads the parameters for the simulations, pertinent among them
% % being x_size (centered at 0 in the x-axis), z_start and z_size.
% % **NOTE:BE CAREFUL!** - params.mat file hasn't been flipped (or augmented or whatever) 
% % because it's not really required any more. labels has been flipped!!
%f0 = 4e6; % Transmit frequecy [Hz]
%fs = 100e6;
%dt = 1/fs;
%params_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_4mhz/dataset/params.mat');
%labels_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_4mhz/dataset/labels.mat');
%channel_data_folder = fullfile(data_prefix,'Dataset_creation_6_fieldII_deterministicGrid_4mhz/dataset/channel_data');
%att_string = 'woatt';

% % Case 1 - WITH ATTENUATION Deterministic grid - 4Mhz, with 100Mhz sampling frequency
%f0 = 4e6; % Transmit frequecy [Hz]
%fs = 100e6;
%dt = 1/fs;
%params_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_4mhz_watt/dataset/params.mat');
%labels_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_4mhz_watt/dataset/labels.mat');
%channel_data_folder = fullfile(data_prefix,'Dataset_creation_6_fieldII_deterministicGrid_4mhz_watt/dataset/channel_data');
%att_string = 'watt';

% % % Case 2 - Deterministic grid - 8Mhz, with 100Mhz sampling frequency
% % The below loads the parameters for the simulations, pertinent among them
% % being x_size (centered at 0 in the x-axis), z_start and z_size.
% % **NOTE:BE CAREFUL!** - params.mat file hasn't been flipped (or augmented or whatever) 
% % because it's not really required any more. labels has been flipped!!
%f0 = 8e6; % Transmit frequecy [Hz]
%fs = 100e6;
%dt = 1/fs;
%params_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_8mhz/dataset/params.mat');
%labels_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_8mhz/dataset/labels.mat');
%channel_data_folder = fullfile(data_prefix,'Dataset_creation_6_fieldII_deterministicGrid_8mhz/dataset/channel_data');
%att_string = 'woatt';

% % Case 3 - WITH ATTENUATION Deterministic grid - 8Mhz, with 100Mhz sampling frequency
%f0 = 8e6; % Transmit frequecy [Hz]
%fs = 100e6;
%dt = 1/fs;
%params_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_8mhz_watt/dataset/params.mat');
%labels_string = fullfile(data_prefix, 'Dataset_creation_6_fieldII_deterministicGrid_8mhz_watt/dataset/labels.mat');
%channel_data_folder = fullfile(data_prefix,'Dataset_creation_6_fieldII_deterministicGrid_8mhz_watt/dataset/channel_data');
%att_string = 'watt';

% % % Case 4 - WITHOUT ATTENUATION Deterministic grid - 4Mhz, with 100Mhz sampling frequency WITH POINT SCATTERERS
% f0 = 4e6; % Transmit frequecy [Hz]
% fs = 100e6;
% dt = 1/fs;
% params_string = fullfile(data_prefix, 'Dataset_creation_7_fieldII_deterministicGrid_4mhz_woatt/dataset/params.mat');
% labels_string = fullfile(data_prefix, 'Dataset_creation_7_fieldII_deterministicGrid_4mhz_woatt/dataset/labels.mat');
% channel_data_folder = fullfile(data_prefix,'Dataset_creation_7_fieldII_deterministicGrid_4mhz_woatt/dataset/channel_data');
% att_string = 'woatt';

% % % Case 5 - WITH ATTENUATION Deterministic grid - 4Mhz, with 100Mhz sampling frequency WITH POINT SCATTERERS
%f0 = 4e6; % Transmit frequecy [Hz]
%fs = 100e6;
%dt = 1/fs;
%params_string = fullfile(data_prefix, 'Dataset_creation_7_fieldII_deterministicGrid_4mhz_watt/dataset/params.mat');
%labels_string = fullfile(data_prefix, 'Dataset_creation_7_fieldII_deterministicGrid_4mhz_watt/dataset/labels.mat');
%channel_data_folder = fullfile(data_prefix,'Dataset_creation_7_fieldII_deterministicGrid_4mhz_watt/dataset/channel_data');
%att_string = 'watt';

% % % Case 6 - WITHOUT ATTENUATION Deterministic grid - 4Mhz, with 100Mhz
% % sampling frequency WITH POINT SCATTERERS and LINE SCATTERERS
% f0 = 4e6; % Transmit frequecy [Hz]
% fs = 100e6;
% dt = 1/fs;
% params_string = fullfile(data_prefix, 'Dataset_creation_8_fieldII_deterministicGrid_4mhz_woatt/dataset/params.mat');
% labels_string = fullfile(data_prefix, 'Dataset_creation_8_fieldII_deterministicGrid_4mhz_woatt/dataset/labels.mat');
% channel_data_folder = fullfile(data_prefix,'Dataset_creation_8_fieldII_deterministicGrid_4mhz_woatt/dataset/channel_data');
% att_string = 'woatt';

% % Case 7 - WITH ATTENUATION Deterministic grid - 4Mhz, with 100Mhz
% sampling frequency WITH POINT SCATTERERS and LINE SCATTERERS
f0 = 4e6; % Transmit frequecy [Hz]
fs = 100e6;
dt = 1/fs;
params_string = fullfile(data_prefix, 'Dataset_creation_8_fieldII_deterministicGrid_4mhz_watt/dataset/params.mat');
labels_string = fullfile(data_prefix, 'Dataset_creation_8_fieldII_deterministicGrid_4mhz_watt/dataset/labels.mat');
channel_data_folder = fullfile(data_prefix,'Dataset_creation_8_fieldII_deterministicGrid_4mhz_watt/dataset/channel_data');
att_string = 'watt';


% #2 - Decide if you want to use IQ or RF as input
input_format = 'IQ';
%input_format = 'RF';

% #3 - Grid sizes and interpolation method - also design the lowpass filter
% and the pulse for lag estimation
downsampling_method = {'interp','imresize','resample'};
chosen_downsampling_method = downsampling_method{1}; % Because scaling it along 2d axis as well

% Also Need to design the lowpass filter for this case
% Designed a lowpass filter with transition band b/w 3 and 4 MHz for 4 MHz tx freq.... 
% Looked at higher frequencies as well. Generalized it to the below.
d = fdesign.lowpass('Fp,Fst,Ap,Ast',f0/2 + 1e6,f0/2 + 2e6,0.5,40,fs); 
Hd = design(d,'equiripple');
% NOTE: Also need to compensate for the delay introduced by this filter!!!
filter_delay = round(mean(grpdelay(Hd))); % Calculate the delay in number of samples. Round it since it might not be an integer.
filter_weights = Hd.Numerator.'; % Hd.Numerator is a row matrix; need to convert it to a column matrix
%     % % Uncomment the below to plot the frequency response of the designed
%     % filter
% 	fvtool(Hd)    

% Also need lag estimation:
% Needed to add lag estimation - verified that the B-mode image is moved
% downwards beyond true position if lag isn’t included and subtracted
pulse_duration = 4;% pulse duration [cycles] - this is set based on what the L3-8 pulse looks like in the transducer spec sheet
pulse = uff.pulse(f0); 
pulse.fractional_bandwidth = 0.65;        % probe bandwidth [1]
t0 = (-1/pulse.fractional_bandwidth/f0): dt : (1/pulse.fractional_bandwidth/f0);
impulse_response = gauspuls(t0, f0, pulse.fractional_bandwidth);
impulse_response = impulse_response-mean(impulse_response); % To get rid of DC

te = (-pulse_duration/2/f0): dt : (pulse_duration/2/f0);
excitation = square(2*pi*f0*te+pi/2);
one_way_ir = conv(impulse_response,excitation);
two_way_ir = conv(one_way_ir,impulse_response);
lag = length(two_way_ir)/2+1;    

% % Case 0 - The tried and tested baseline
grid_x = 128;
grid_z = 256;

% % Case 1 - larger image size to hopefully do better
% grid_x = 512;
% grid_z = 640;

% Case 2 - larger but not so aggressively so
%grid_x = 256;
%grid_z = 384;

% #4 - Specify whether to add random noise augmentation
% % Case 0 - The tried and tested baseline case - no random noise
% augmentation
noise_snr = Inf;
% save_dir = 'processed_data';
% Case 1 - With random noise augmentatioin
% noise_snr = 3; % In dB

% #5 - Specify directory to store processed data - linked with the
% tx frequency, input type, downsampling idea and grid size and noise SNR
%save_dir = sprintf('processed_data_%dMHz_%s_%s_%s_grid_%d_%d_snr_%d',f0/1e6,...
%    att_string, input_format,  chosen_downsampling_method,grid_z,grid_x,noise_snr);
% save_dir = sprintf('processed_data_%dMHz_%s_%s_%s_grid_%d_%d_snr_%d_onlyANNoHyp_WPOINTSCATTERERS',f0/1e6,...
%     att_string, input_format, chosen_downsampling_method,grid_z,grid_x,noise_snr);
save_dir = sprintf('processed_data_%dMHz_%s_%s_%s_grid_%d_%d_snr_%d_onlyANNoHyp_WPOINTANDLINESCATTERERS',f0/1e6,...
    att_string, input_format, chosen_downsampling_method,grid_z,grid_x,noise_snr);
sprintf('Outputting files to %s',save_dir)

%% Load parameter settings for the simulated data
load(params_string);
load(labels_string);

%% Create directory to store processed data in case it hasn't been created yet
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% Load the simulation files

% Start timing
tic;

% start parallel pool if not already running
p = gcp;
 
disp('Now saving processed data..............');
sprintf('%d data samples to process..........',length(labels))
parforProgress(length(labels));
parfor parfor_idx=1:length(labels)
% for parfor_idx=1:length(labels)
%parfor parfor_idx=22231:33345
    %----------------------------------------------------------------------
    % Load the raw data
    %----------------------------------------------------------------------
    inner_loop_channel_data_folder = channel_data_folder;
    channel_data_file = sprintf([inner_loop_channel_data_folder '/%06d.mat'], parfor_idx-1);
    data_cell = load(channel_data_file);
    channel_data_wonoise = data_cell.v_2;
    
    %----------------------------------------------------------------------
    % Set probe characteristics
    %----------------------------------------------------------------------
    % f0 (transmit frequency) and fs (sampling frequency) have already been set prior
    c = data_cell.curr_label{1}.c;    % Speed of sound [m/s]
    N_elements = 128;                 % Number of probe elements.
    pitch = 0.3e-3;                   % Pitch of the probe [m]

    % Note: On doing the math, you get array size is in total
    %                       $$(N-1)*pitch$$
    % from center of leftmost element to center of right most element
    % (makes sense).
    x_in_grid_points = linspace(-(N_elements-1)*pitch/2, (N_elements-1)*pitch/2, N_elements);
    x_out_grid_points = linspace(-(N_elements-1)*pitch/2, (N_elements-1)*pitch/2, grid_x);
    %----------------------------------------------------------------------
    % Associate axial samples to sample times, and z coordinates - basically z = ct/2 formula    
    %----------------------------------------------------------------------      
    min_sample = data_cell.t_2*fs;
    t_in = ((0:size(channel_data_wonoise,1)-1)+min_sample).'/fs - lag*dt; % spent a long time convincing myself that i'm not missing a factor of 2 and am in fact staying true to the ntnu doc
    z_in=t_in*c/2; 
    z_out = linspace(z_start,z_start+z_size,grid_z);
    [~, z_in_relevant_start_idx] = min(abs(z_in-z_start));
    [~, z_in_relevant_end_idx]   = min(abs(z_in-(z_start+z_size)));
    z_in_relevant = z_in(z_in_relevant_start_idx:z_in_relevant_end_idx);
    
    %----------------------------------------------------------------------
    % NOTE: http://www.ultrasonix.com/wikisonix/index.php/Ultrasound_Image_Computation - nice diagram illustrating steps from RF To IQ to B-Mode
    %----------------------------------------------------------------------

    %----------------------------------------i-----------------------------
    % if save_dir = 'processed_data' i.e. direct downsampling with aliasing
    %----------------------------------------------------------------------
    if strcmp(input_format, 'RF')
        %----------------------------------------------------------------------
        % Downsample the channel data directly without any of the other
        % steps
        %----------------------------------------------------------------------
        if strcmp(chosen_downsampling_method,'interp') % No antialising filter
            % Running the interpolation on the GPU is a lot slower
            % Spline interpolation is also a lot slower
            [X_in_mesh,Z_in_mesh] = meshgrid(x_in_grid_points,z_in);
            [X_out_mesh,Z_out_mesh] = meshgrid(x_out_grid_points,z_out);
            v_2_z_out = interp2(X_in_mesh,Z_in_mesh,channel_data_wonoise,X_out_mesh,Z_out_mesh,'linear',0);
        elseif strcmp(chosen_downsampling_method,'imresize') % Has an antialiasing filter by default
            % imresize is slightly faster than linear interpolation - 8s to 5s
            v_2_z_out = imresize(channel_data_wonoise(z_in_relevant_start_idx:z_in_relevant_end_idx,:),[length(z_out), length(x_out_grid_points)]); 
        elseif strcmp(chosen_downsampling_method,'resample')
            % Uses a polyphase anti-aliasing filter. If input is a matrix, then 
            % resample treats each column as an independent channel.
            % resample() is faster than imresize - 5s to 2s
            % NOTE: This seems the most accurate - frequency content most 
            % closely resembles the highly sampled signal
            v_2_z_out = resample(channel_data_wonoise(z_in_relevant_start_idx:z_in_relevant_end_idx,:), length(z_out), length(z_in_relevant));
            v_2_z_out  = resample(v_2_z_out.', length(x_out_grid_points), length(x_in_grid_points)).';
        end 
        %----------------------------------------------------------------------
        % Do random noise augmentation and normalize the data
        %----------------------------------------------------------------------    
        v_2_z_out_noiseless  = v_2_z_out;
        v_2_z_out = awgn(v_2_z_out, noise_snr, 'measured');
        % Normalize the data
        v_2_z_out = v_2_z_out/(max(abs(v_2_z_out(:))));
        
        %----------------------------------------------------------------------
        %     % % Uncomment the below to plot frequency content of the downsampled signal
        %     slice = v_2_z_out(:,64);
        %     plot_freq_content(slice,fs*length(z_out)/length(z_in_relevant));    
        %     xlim([-fs/2 fs/2]);
        %     imagesc((abs(v_2_z_out)), [0 1]); colormap gray
        %     imagesc(20*log10(abs(v_2_z_out)), [-60 0]); colormap gray
        v_2_z_out = reshape(v_2_z_out, 1, size(v_2_z_out,1), size(v_2_z_out,2));
        
        
    %----------------------------------------i-----------------------------
    % if save_dir = 'processed_data_demodulation' i.e. demodulated downsampling
    %----------------------------------------------------------------------        
    elseif strcmp(input_format,'IQ')
        %----------------------------------------i-------------------------
        % I - Raw RF data
        %------------------------------------------------------------------
    %     % Uncomment the below to plot frequency content of the raw RF signal
    %     for idx_temp = 1:grid_x
    %         slice = channel_data_wonoise(:,idx_temp);
    %         plot_freq_content(slice,fs);
    %         xlim([-2*f0 2*f0]);
    %         pause(1);        
    %         close all;
    %     end

        %------------------------------------------------------------------
        % II - Downmixed data - multiply raw RF data with a complex 
        % exponential to bring it down in the frequency domain
        %------------------------------------------------------------------
        complex_sinusoid = exp(-1i*2*pi*f0*t_in); %Section 3.3, Pg 6 of http://folk.ntnu.no/htorp/Undervisning/TTK10/IQdemodulation.pdf
        v_2_downmixed = channel_data_wonoise .* complex_sinusoid; % Section 3.3, Pg 6 - Note how the equivalent operation can be done with a cosine and a sine wave in that text... 
    %     % Uncomment the below to plot frequency content of the downmixed signal
    %     slice = v_2_downmixed(:,64);
    %     plot_freq_content(slice,fs);

        %------------------------------------------------------------------
        % III - Lowpass filter the downmixed data
        %------------------------------------------------------------------
        % NOTE:https://www.mathhworks.com/help/matlab/ref/filter.html - If x 
        % is a matrix, then filter acts along the first dimension and returns 
        % the filtered data for each column.        
        % 
        % v_2_downmixed_lpf = filter(Hd,v_2_downmixed); % NOTE: parfor throws a hissy fit if you try using filter. Instead I did it manually
        % 
        % v_2_downmixed_lpf = conv2(v_2_downmixed,filter_weights,'full');
        % v_2_downmixed_lpf = v_2_downmixed_lpf(1:size(v_2_downmixed,1),:); % Remove the outputs towards the tail. Replicating the behavior of using filter(Hd, v_2_downmixed);
        %******************************************************************
        % NOTE: This introduces a delay though! Need to compensate for
        % it... Important thing I caught that probably makes no difference
        % in the result whatsoever :| So finding the group delay and
        % compensating for it as in
        % https://www.mathworks.com/help/signal/ug/compensate-for-the-delay-introduced-by-an-fir-filter.html
        % is verified to be equivalent to just doing 'same in conv2
        %******************************************************************
        v_2_downmixed_lpf = conv2(v_2_downmixed,filter_weights,'same');
%         c = filter(Hd,v_2_downmixed);
%         c(1:filter_delay,:)=[]; % c is the same as what I got using conv2 with the 'same' option
        
        % From section 3.4, Pg 7 of ntnu doc - "The low-pass filter on the
        % complex signal can be thought of as a filter applied to the real
        % andimaginary  part  separately" - so think you can manipulate both
        % independently... and justifies doing so...
        % Verified this above by low pass filtering real and imaginary parts of
        % v_2_downmixed separately and then adding them. Same result as doing
        % it directly.
        %     % Uncomment the below to plot frequency content of the downmixed lowpass filtered signal
        %     slice = v_2_downmixed_lpf(:,64);
        %     plot_freq_content(slice,fs);
        %     xlim([-fs*length(z_out)/length(z_in_relevant)/2 fs*length(z_out)/length(z_in_relevant)/2]);

        %------------------------------------------------------------------
        % IV - Downsample the data
        %------------------------------------------------------------------
        if strcmp(chosen_downsampling_method,'interp') % No antialising filter
            % Running the interpolation on the GPU is a lot slower
            % Spline interpolation is also a lot slower
            [X_in_mesh,Z_in_mesh] = meshgrid(x_in_grid_points,z_in);
            [X_out_mesh,Z_out_mesh] = meshgrid(x_out_grid_points,z_out);
            v_2_z_out = interp2(X_in_mesh,Z_in_mesh,v_2_downmixed_lpf,X_out_mesh,Z_out_mesh,'linear',0);            
        elseif strcmp(chosen_downsampling_method,'imresize') % Has an antialiasing filter by default
            % imresize is slightly faster than linear interpolation - 8s to 5s
            v_2_z_out = imresize(v_2_downmixed_lpf(z_in_relevant_start_idx:z_in_relevant_end_idx,:),[length(z_out), length(x_out_grid_points)]);
        elseif strcmp(chosen_downsampling_method,'resample')
            % Uses a polyphase anti-aliasing filter. If input is a matrix, then 
            % resample treats each column as an independent channel.
            % resample() is faster than imresize - 5s to 2s
            % NOTE: This seems the most accurate - frequency content most 
            % closely resembles the highly sampled signal
            v_2_z_out = resample(v_2_downmixed_lpf(z_in_relevant_start_idx:z_in_relevant_end_idx,:), length(z_out), length(z_in_relevant));
            v_2_z_out = resample(v_2_z_out.', length(x_out_grid_points), length(x_in_grid_points)).';            
        end
        %----------------------------------------------------------------------
        % Do random noise augmentation and normalize the data
        %----------------------------------------------------------------------    
        v_2_z_out_noiseless  = v_2_z_out;
        v_2_z_out = awgn(v_2_z_out,noise_snr,'measured');
%         % Uncomment the below to plot frequency content of the downsampled signal and what the data looks like itself
%         slice = v_2_z_out(:,64);
%         plot_freq_content(slice,fs*length(z_out)/length(z_in_relevant));
%         xlim([-fs*length(z_out)/length(z_in_relevant)/2 fs*length(z_out)/length(z_in_relevant)/2]);
%         imagesc((abs(v_2_z_out)), [0 1]); colormap gray
%         imagesc(20*log10(abs(v_2_z_out)), [-60 0]); colormap gray

        % Normalize the data
        v_2_z_out = v_2_z_out/(max(abs(v_2_z_out(:))));

        %------------------------------------------------------------------
        % V - Split the complex IQ data into real and imaginary components 
        % and add the extra channel dimension at the start and concatenate 
        % them to get a 2 channel RF image
        %------------------------------------------------------------------
        re_v_2_z_2 = real(v_2_z_out); re_v_2_z_2 = reshape(re_v_2_z_2, 1, size(re_v_2_z_2,1), size(re_v_2_z_2,2));
        im_v_2_z_2 = imag(v_2_z_out); im_v_2_z_2 = reshape(im_v_2_z_2, 1, size(im_v_2_z_2,1), size(im_v_2_z_2,2));
        v_2_z_out  = cat(1,re_v_2_z_2,im_v_2_z_2);
    end

    %----------------------------------------------------------------------
    % Prepare the Ground Truth Segmentation Mask
    % Code should be easily modified to multiple cysts
    %----------------------------------------------------------------------
    [X,Z] = meshgrid(x_out_grid_points,z_out);
    GT_z_out = ( ((X-data_cell.curr_label{1}.x).^2 +...
        (Z-data_cell.curr_label{1}.z).^2) < data_cell.curr_label{1}.r^2);
    GT_z_out = reshape(GT_z_out ,1,size(GT_z_out,1), size(GT_z_out,2));

    %----------------------------------------------------------------------
    % Create the curr_label variable
    %----------------------------------------------------------------------        
    temp_curr_label = zeros(1,5);
    temp_curr_label(1) = data_cell.curr_label{1}.c;
    temp_curr_label(2) = data_cell.curr_label{1}.amp;
    temp_curr_label(3) = data_cell.curr_label{1}.r;
    temp_curr_label(4) = data_cell.curr_label{1}.z;
    temp_curr_label(5) = data_cell.curr_label{1}.x;

    %----------------------------------------------------------------------
    % Create the OG DAS B-mode image
    %----------------------------------------------------------------------            
    b_data = beamform(inner_loop_channel_data_folder,parfor_idx-1,temp_curr_label,fs,f0, x_out_grid_points, z_out);
    b_image = reshape(b_data.data, length(z_out), length(x_out_grid_points));
    chosen_B_image = 20*log10(abs(b_image./max(abs(b_image(:)))));
    chosen_B_image(chosen_B_image<=-60)=-60;
    chosen_B_image = (chosen_B_image+60)*255/60;    
    
    %----------------------------------------------------------------------
    % Below is code for the contrast enhancement idea I had, where you try
    % to train it to generate the "true" contrast
%     chosen_B_image_exp = 10.^(chosen_B_image/20);
%     background_image_exp = zeros(size(chosen_B_image_exp));
%     background_image_exp(logical(1-squeeze(GT_z_out))) = chosen_B_image_exp(logical(1-squeeze(GT_z_out)));
%     mean_background_image = sum(background_image_exp(:))/sum(1-logical(squeeze(GT_z_out(:))));
%     mean_in_cyst = mean_background_image*data_cell.curr_label{1}.amp;
%     chosen_B_image_exp(logical(squeeze(GT_z_out))) = mean_in_cyst;
%     enhanced_B_image = 20*log10(chosen_B_image_exp);
%     enhanced_B_image(enhanced_B_image<=-60)=-60;
%     enhanced_B_image = (enhanced_B_image+60)*255/60;
%     enhanced_B_image = uint8(reshape(enhanced_B_image,1,size(enhanced_B_image,1),size(enhanced_B_image,2)));
    
    %----------------------------------------------------------------------
    % Create the enhanced (contrast-enhancement) DAS B-mode image
    %----------------------------------------------------------------------            
    enhanced_B_image = chosen_B_image;
    enhanced_B_image(logical(squeeze(GT_z_out)))=0;
    chosen_B_image = uint8(reshape(chosen_B_image,1,size(chosen_B_image,1),size(chosen_B_image,2)));
    enhanced_B_image = uint8(reshape(enhanced_B_image,1,size(enhanced_B_image,1),size(enhanced_B_image,2)));
    
    %----------------------------------------------------------------------
    % Save the data to file
    %----------------------------------------------------------------------
    out_channel_data_file = fullfile(save_dir, sprintf('%06d.mat',parfor_idx-1));
    real_flag = 0; % As the data is simulated
    parsave(out_channel_data_file, GT_z_out, v_2_z_out, chosen_B_image, enhanced_B_image, parfor_idx-1, temp_curr_label, real_flag);
    parforProgress;
end
parforProgress(0);
disp('Processing done!');

% kill parallel pool
delete(p);

% Total time
time = toc;
sprintf('Total time taken for data pre-processing is %d seconds', time)

