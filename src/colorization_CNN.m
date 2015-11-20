%function test_example_CNN
addpath(genpath('DeepLearnToolbox'));
train_flag = 1
data_path = '/home/jiangliang/code/semantic-colorization/data/yuv_image/';
params_path = '/home/jiangliang/code/semantic-colorization/params/';
disp('Loading data...');
tic
%load([data_path 'image_y']);
%load([data_path 'image_uv']);

load([data_path 'mini_image_y']);
load([data_path 'mini_image_uv']);
toc
%
%train_x = double(image_y(:, :, 1 : 1));
%test_x = double(image_y(:, :, 1001 : 1001));
%train_y = double(image_uv(:, 1 : 1));
%test_y = double(image_uv(:, 1001 : 1001));
disp('Splitting and Shufffling data...');
num_train_samples = 496;
num_test_samples = 4;
tic
train_x = double(mini_image_y(:, :, 1 : num_train_samples));
test_x = double(mini_image_y(:, :, num_train_samples + 1 : num_train_samples + num_test_samples));
train_y = double(mini_image_uv(:, 1 : num_train_samples));
test_y = double(mini_image_uv(:, num_train_samples + 1 : num_train_samples + num_test_samples));
order = randperm(size(train_y, 2));

train_x = train_x(:, :, order);
train_y = train_y(:, order);
toc

%Split into patches
row_patches = 8;
col_patches = 8;

[train_x_patches, train_y_patches] = split_data(train_x, train_y, row_patches, col_patches);
[test_x_patches, test_y_patches] = split_data(test_x, test_y, row_patches, col_patches);
disp(['training samples: ' num2str(size(train_y_patches, 2)) ', testing samples: ' num2str(size(test_y_patches, 2))]);

%train_x = double(reshape(train_x',28,28,60000))/255;
%test_x = double(reshape(test_x',28,28,10000))/255;
%train_y = double(train_y');
%test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
if (train_flag)
    rand('state',0)
    tic
    disp('Setting up CNN...');
    %cnn.layers = {
    %    struct('type', 'i') %input layer
    %    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5) %convolution layer
    %    struct('type', 's', 'scale', 3) %sub sampling layer
    %    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5) %convolution layer
    %    struct('type', 's', 'scale', 2) %subsampling layer
    %    struct('type', 'c', 'outputmaps', 48, 'kernelsize', 5) %convolution layer
    %    struct('type', 's', 'scale', 2) %subsampling layer
    %    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5) %convolution layer
    %    struct('type', 's', 'scale', 2) %subsampling layer
    %};

    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 1) %sub sampling layer
        struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 1) %sub sampling layer
        struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };
    
    
    opts.alpha = 10e-5;
    opts.batchsize = 128;
    opts.numepochs = 2;
    opts.count = 1;
    opts.c = 20000;

    cnn = cnnsetup(cnn, train_x_patches, train_y_patches);
    toc
    disp('Training CNN...');
    tic
    cnn = cnntrain(cnn, train_x_patches, train_y_patches, opts);
    toc
    now_time = datestr(now, 'yyyy-mm-DD-HHMM');
    save([params_path 'cnn.' now_time], 'cnn', '-v7.3');
else
    %load newest parameter
    disp('Loading CNN')
    tic
    cnn_params = dir(params_path);
    load([params_path cnn_params(length(cnn_params)).name]);
    toc
end

tic
disp('Testing CNN...');
[er, bad] = cnntest(cnn, test_x, test_y);
toc

%plot mean squared error
%figure; plot(cnn.rL);
%assert(er<0.12, 'Too big error');
