function show_colorization_results()
addpath(genpath('DeepLearnToolbox'));
disp('Loading results data');
result_path = '/home/jiangliang/code/semantic-colorization/result/';
result_file = dir([result_path 'result*']);
original_file = dir([result_path 'original*']);
plot_flag = 0;

result_file(end).name
original_file(end).name
load([result_path result_file(end).name]);
load([result_path original_file(end).name]);
images = size(result_yuv, 4);
if images > 4
    images = 4;
end
size(result_yuv)
size(original_yuv)
samples_to_display = randperm(size(result_yuv, 4));
samples_to_display = samples_to_display(1 : images)
disp('displaying');
for i = 1 : length(samples_to_display)
    original_rgb_image = yuv2rgb(original_yuv(:, :, :, samples_to_display(i)), plot_flag);
    result_rgb_image = yuv2rgb(result_yuv(:, :, :, samples_to_display(i)), plot_flag);
    figure();
    subplot(1, 2, 1);
    title('original');
    imshow(original_rgb_image);
    subplot(1, 2, 2);
    title('colorization');
    imshow(result_rgb_image);
end
