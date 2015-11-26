data_path = '/home/jiangliang/code/semantic-colorization/data/yuv_image/';
disp('Loading data...');
k = 128;
max_iter = 10000;
load([data_path 'mini_uv.mat']);
image_size = size(mini_uv, 1) / 2;
temp = reshape(mini_uv, image_size, 2, size(mini_uv, 2));
uv = zeros(size(temp, 1) * size(temp, 3) , 2);
for i = 1 : size(temp, 3)
    uv((i - 1) * image_size + 1 : i * image_size, :) = temp(:, :, i);
end
opts = statset('MaxIter', max_iter);
disp('Clustering');
tic
[idx, c] = kmeans(uv, k, 'Replicates', 5, 'EmptyAction', 'singleton', 'Options', opts);
toc
idx = reshape(idx, [size(idx, 1) / size(mini_uv, 2), size(mini_uv, 2)]);
save([data_path 'mini_class.mat'], 'idx', '-v7.3');
save([data_path 'mini_cen.mat'], 'c', '-v7.3');
