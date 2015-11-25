%function test_example_CNN
addpath(genpath('DeepLearnToolbox'));
train_flag = 1
data_path = '/home/jiangliang/code/semantic-colorization/data/yuv_image/';
params_path = '/home/jiangliang/code/semantic-colorization/params/';
tic
%load([data_path 'image_y']);
%load([data_path 'image_uv']);
if ~ exist('mini_image_y') || ~exist('mini_image_uv')
    disp('Loading data...');
    load([data_path 'mini_image_y']);
    load([data_path 'mini_image_uv']);
end
toc
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

%Split into patches

patch.row_size = 45;
patch.col_size = 45;
patch.num = 1024;
%
%
%
disp('Augmenting data...')
tic
[aug_train_x, aug_train_y] = augment(train_x, train_y, patch);
[aug_test_x, aug_test_y] = augment(test_x, test_y, patch);
toc

disp('Fetch random patches from training set...');
tic
[train_x_patches, train_y_patches] = rand_patches(train_x, train_y, patch);
toc
disp('Splitting testing data')
tic
%[test_x_patches, test_y_patches] = split_test_data(aug_test_x, aug_test_y, patch);
toc
%
%aug_test_x = zeros(size(test_x, 1) + 2 * (size(test_x, 1) / row_patches - 2, size(test_x, 2) + 2 * (size(test_x, 2) / col_patches - 2, size(test_x, 3));
%aug_test_x(row_patches / 2 : end - row_patches / 2 + 1, col_patches / 2 : end - col_patches / 2 + 1, :) = test_x;
%row_num = size(x, 1) / 2;
%col_num = size(x, 2) / 2;
%for i = row_patches / 2 : 2 : size(aut_test_x, 1) - row_patches / 2 + 1
%    for j = col_patches / 2 : 2 : size(aut_test_x, 1) - col_patches / 2 + 1
%        test_x_patches(:, :, (i - 1) * row_num + j) = aug_test_x(i - patch_row_size / 2 + 1 : i + patch_row_size / 2, j - patch_col_size / 2 + 1, j + patch_col_size);
%    end
%end

%if ~exist('patch_x') || ~exist('patch_y')
%    disp('Splitting testing data');
%    tic
%    patch_x = zeros(patch_row_size, patch_col_size, num_row * num_col, size(test_y, 2));
%    patch_y = zeros(8, num_row * num_col, size(test_y, 2));
%    for l = 1 : size(test_x, 3)
%        for i = 1 : num_row
%            for j = 1 : num_col
%                patch_x(:, :, (i - 1) * num_row + j, l) = test_x((i - 1) * 2 + 1 : (i - 1) * 2 + patch_row_size, (j - 1) * 2 + 1 : (j - 1) * 2 + patch_col_size, l);
%                test_u = reshape(test_y(1 : size(test_y, 1) / 2, l), size(test_x, 1), size(test_x, 2));
%                test_v = reshape(test_y(size(test_y, 1) / 2 + 1 : end, l), size(test_x, 1), size(test_x, 2));
%                row_start = (i - 1) * 2 + patch_row_size / 2;
%                row_end = (i - 1) * 2 + 1 + patch_row_size / 2;
%                col_start = (j - 1) * 2 + patch_col_size / 2;
%                col_end = (j - 1) * 2 + 1 + patch_col_size / 2; 
%                u_patch = test_u(row_start : row_end, col_start : col_end);
%                v_patch = test_v(row_start : row_end, col_start : col_end);
%                
%                patch_y(:, (i - 1) * num_row + j, l) = [u_patch(:);
%                                                     v_patch(:)];
%            end
%        end
%    end
%    toc
%end

train_y_patches(1 : size(train_y_patches, 1) / 2, :) = train_y_patches(1 : size(train_y_patches, 1) / 2, :)  * 2.294;
train_y_patches(size(train_y_patches, 1) / 2 + 1 : end, :) = train_y_patches(size(train_y_patches, 1) / 2 + 1 : end, :)  * 1.626;

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
opts.alpha = 10e-5;
opts.batchsize = 64;
opts.numepochs = 100;
opts.count = 1;
opts.c = 20000;
opts.activation_type = 'sigmoid';
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
%for i = 1 : size(patch_x, 4)
%    [er, bad] = cnntest(cnn, test_x(:, :, i), patch_x(:, :, :, i), patch_y(:, :, :, i), opts);
%end

%[er, bad] = cnntest(cnn, test_x_patches, test_y_patches);
toc

%plot mean squared error
%figure; plot(cnn.rL);
%assert(er<0.12, 'Too big error');
quit;
