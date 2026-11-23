clc; close all; clear

%% Load hippocampus and cortex data
Sleep_data = load("C:\Users\mdenevesdefons\OneDrive - Delft University of Technology\Documents\PhD\Rafael Data 2\A07 Rafael Pedrosa\block4_sleep_OE.mat");

CTX_data_path = "C:\Users\mdenevesdefons\OneDrive - Delft University of Technology\Documents\PhD\Rafael Data 2\A07 Rafael Pedrosa\block444999fr_50Hz_241x384_Bandpass_Cheby_high_0.5FHz_D_F0_.raw";

% Read file
fid = fopen(CTX_data_path, 'r', 'ieee-be');
raw_video = fread(fid, "float32=>float32");
fclose(fid);

%% Create and plot hippocampus lfp dataset
for i=1:size(Sleep_data.Data_OE.HPC_LFP,2)
    HPC_LFP{i} = downsample(Sleep_data.Data_OE.HPC_LFP{i},20); %originally 20kHz ---> downsample to 1kHz
end
t = size(HPC_LFP{1},1);
fs = 1000;

fd = [49 99 149 199 249 299 349];
fu = [51 101 151 201 251 301 351];

for i=1:size(fd,2)
    d(i,:) = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',fd(i),'HalfPowerFrequency2',fu(i), ...
               'DesignMethod','butter','SampleRate',fs);
end

% filter signal 
for i=1:size(HPC_LFP,2)
    HPC_LFP_filtered{i} = HPC_LFP{i};
    for j=1:size(fd,2)
        HPC_LFP_filtered{i} = filtfilt(d(j,:), HPC_LFP_filtered{i});
    end
end

figure;
subplot(2,1,1)
spectrogram(detrend(zscore(HPC_LFP{11})),fs,0.9*fs,2^14,fs)
subtitle('Unfiltered')
subplot(2,1,2)
spectrogram(detrend(zscore(HPC_LFP_filtered{11})),fs,0.9*fs,2^14,fs)
subtitle('Filtered')

for i=1:size(HPC_LFP,2)
    HPC_LFP_filtered_normalized{i} = detrend(zscore(HPC_LFP_filtered{i}));
end

figure;
for i=1:size(HPC_LFP_filtered_normalized,2)
    plot(0:1/fs:(t-1)/fs,HPC_LFP_filtered_normalized{i}-i*ones(t,1)*1000)
    hold on;
end
xlabel('Time (s)')
ylabel('LFP/channel')
%% Create and plot cortex voltage imaging dataset
% Convert into frames
W  = 241;      % frame width
H = 384;      % frame height
frameFreq = 50;

numFrames = length(Sleep_data.Data_OE.frames);
sleep_video = reshape(raw_video, [W, H, numFrames]);
sleep_video_half1 = permute(sleep_video(:,:,1:20000), [2 1 3]);
sleep_video_half2 = permute(sleep_video(:,:,20001:end), [2 1 3]);

figure;
imagesc(sleep_video_half1(:,:,1000));
colorbar;
