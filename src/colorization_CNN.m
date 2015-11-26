%function test_example_CNN
addpath(genpath('DeepLearnToolbox'));
train_flag = 1;
data_path = '/home/jiangliang/code/semantic-colorization/data/yuv_image/';
params_path = '/home/jiangliang/code/semantic-colorization/params/';
result_path = '/home/jiangliang/code/semantic-colorization/result/';
tic
%load([data_path 'image_y']);
%load([data_path 'image_uv']);
if ~ exist('mini_image_y') || ~exist('mini_image_uv')
    disp('Loading data...');
    load([data_path 'mini_image_y']);
    load([data_path 'mini_image_uv']);
end
toc

opts.alpha = 10e-5;
opts.batchsize = 64;
opts.numepochs = 200
opts.count = 1;
opts.c = 10000
opts.activation_type = 'sigmoid';
opts.row_size = 45;
opts.col_size = 45;
opts.num = 1024;

%
%train_x = double(image_y(:, :, 1 : 1));
%test_x = double(image_y(:, :, 1001 : 1001));
%train_y = double(image_uv(:, 1 : 1));
%test_y = double(image_uv(:, 1001 : 1001));
disp('Splitting and Shufffling data...');
num_train_samples = 20;
num_test_samples = 2;
tic
train_x = double(mini_image_y(:, :, 1 : num_train_samples));
test_x = double(mini_image_y(:, :, num_train_samples + 1 : num_train_samples + num_test_samples));
train_y = double(mini_image_uv(:, 1 : num_train_samples));
test_y = double(mini_image_uv(:, num_train_samples + 1 : num_train_samples + num_test_samples));
order = randperm(size(train_y, 2));

train_x = train_x(:, :, order);
train_y = train_y(:, order);
toc

%yuv = zeros(256, 256, 3);
%yuv(:, :, 1) = test_x(:, :, 2);
%yuv(:, :, 2 : 3) = reshape(test_y(:, 2), 256, 256, 2);
%rgb = yuv2rgb(yuv, 0);
%figure();
%imshow(rgb)
%Split into patches

%
%
%
disp('Augmenting data...')
tic
[aug_train_x, aug_train_y] = augment(train_x, train_y, opts);
[aug_test_x, aug_test_y] = augment(test_x, test_y, opts);

%yuv = zeros(300, 300, 3);
%yuv(:, :, 1) = aug_test_x(:, :, 2);
%yuv(:, :, 2 : 3) = reshape(aug_test_y(:, 2), 300, 300, 2);
%rgb = yuv2rgb(yuv, 0);
%figure();
%imshow(rgb);
toc

disp('Fetch random patches from training set...');
tic
[train_x_patches, train_y_patches] = rand_patches(train_x, train_y, opts);
toc
if ~exist('test_x_patches') || ~exist('test_y_patches')
    disp('Splitting testing data')
    tic
    [test_x_patches, test_y_patches] = split_test_data(aug_test_x, aug_test_y, opts);
    toc
end

train_y_patches(1 : size(train_y_patches, 1) / 2, :) = train_y_patches(1 : size(train_y_patches, 1) / 2, :)  * 2.294;
train_y_patches(size(train_y_patches, 1) / 2 + 1 : end, :) = train_y_patches(size(train_y_patches, 1) / 2 + 1 : end, :)  * 1.626;

test_y_patches(1 : size(test_y_patches, 1) / 2, :) = test_y_patches(1 : size(test_y_patches, 1) / 2, :)  * 2.294;
test_y_patches(size(test_y_patches, 1) / 2 + 1 : end, :) = test_y_patches(size(test_y_patches, 1) / 2 + 1 : end, :)  * 1.626;

%order = randperm(size(train_y_patches, 2));
%train_x_patches = train_x_patches(:, :, order);
%train_y_patches = train_y_patches(:, order);

disp(['training samples: ' num2str(size(train_y_patches, 2))]);

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
        struct('type', 'c', 'outputmaps', 16, 'kernelsize', 7) %convolution layer
        struct('type', 's', 'scale', 3) %sub sampling layer
        struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 1) %sub sampling layer
        struct('type', 'c', 'outputmaps', 48, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 1) %sub sampling layer
    };
    
    

    cnn = cnnsetup(cnn, train_x_patches, train_y_patches);
    toc
    disp('Training CNN...');
    tic
    cnn = cnntrain(cnn, train_x_patches, train_y_patches, opts);
    toc
    now_time = datestr(now, 'yyyy-mm-DD-HHMM');
    save([params_path 'cnn.' now_time '.mat'], 'cnn', '-v7.3');
else
    %load newest parameter
    tic
    cnn_params = dir(params_path);
    disp(['Loading ' cnn_params(end).name]);
    load([params_path cnn_params(end).name]);
    toc
end
    
tic

%disp('Testing CNN...');
original_yuv = zeros(size(train_x, 1) , size(train_x, 2), 3, size(test_x_patches, 4));
result_yuv = zeros(size(train_x, 1), size(train_x, 2), 3, size(test_x_patches, 4));
size(original_yuv)
size(result_yuv)
for i = 1 : size(test_x_patches, 4)
    [er, original, result] = cnntest(cnn, test_x(:, :, i), test_x_patches(:, :, :, i), test_y_patches(:, :, i), opts);
    original(:, :, 2) = original(:, :, 2) / 2.294;
    original(:, :, 3) = original(:, :, 3) / 1.626;
    original_yuv(:, :, :, i) = original;
    
    result(:, :, 2) = result(:, :, 2) / 2.294;
    result(:, :, 3) = result(:, :, 3) / 1.626;
    result_yuv(:, :, :, i) = result;
end
now_time = datestr(now, 'yyyy-mm-DD-HHMM');
save([result_path 'result_yuv.' now_time '.mat'], 'result_yuv', '-v7.3');
save([result_path 'original_yuv.' now_time '.mat'], 'original_yuv', '-v7.3');

%[er] = cnntest(cnn, test_x_patches, test_y_patches);
toc
disp(['error' num2str(er)]);

%plot mean squared error
%figure; plot(cnn.rL);
%assert(er<0.12, 'Too big error');
quit
