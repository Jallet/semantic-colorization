addpath(genpath('../DeepLearnToolbox'));
batch_x = rand(28,28,5);
batch_y = randint(1,5, [1 10]);

opts.alpha = 10e-5;
opts.batchsize = 5;
opts.numepochs = 30;
opts.count = 1;
opts.c = 500;
opts.activation_type = 'relu';
opts.row_size = 45;
opts.col_size = 45;
opts.num = 1024;
opts.classes = 128

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, batch_x, batch_y, opts);

cnn = cnnff(cnn, batch_x, opts);
cnn = cnnbp(cnn, batch_y, opts);
cnnnumgradcheck(cnn, batch_x, batch_y, opts);
