original_image_path = '/home/jiangliang/datasets/flicker8k/flicker_image/Flicker8k_Dataset/';
resized_image_path = '/home/jiangliang/datasets/flicker8k/flicker_image/flicker8k_resized_image/';
yuv_image_path = '/home/jiangliang/code/semantic-colorization/data/yuv_image/';
resized_width = 256;
resized_height = 256;
plot_flag = 0;

original_images = dir(original_image_path);
image_y = zeros(resized_height, resized_width, length(original_images) - 2);
image_uv = zeros(2 * resized_height * resized_width, length(original_images) - 2);
for i = 3 : length(original_images)
    image = imread([original_image_path, original_images(i).name]);
    resized_image = imresize(image, [resized_height, resized_width]);
    yuv_image = rgb2yuv(resized_image, plot_flag);
    image_y(:, :, i - 2) = yuv_image(:, :, 1);
    image_uv(:, i - 2) = [reshape(yuv_image(:, :, 2), resized_width * resized_height, 1); 
                            reshape(yuv_image(:, :, 3), resized_width * resized_height, 1)];
    imwrite(resized_image, [resized_image_path, 'resized_', original_images(i).name]);
end
save([yuv_image_path, 'image_y'], 'image_y', '-v7.3');
save([yuv_image_path, 'image_uv'], 'image_uv', '-v7.3');
