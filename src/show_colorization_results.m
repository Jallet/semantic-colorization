function show_colorization_results(patches)
disp('Loading results data');
load('/home/jiangliang/code/semantic-colorization/result/original_yuv.mat')
load('/home/jiangliang/code/semantic-colorization/result/result_yuv.mat')
samples_to_display = randperm(size(result_yuv, 4) / ((patches) .^ 2));
samples_to_display = samples_to_display(1 : 4);
for i = 1 : length(samples_to_display)
    original_patches = original_yuv(:, :, :, ((samples_to_display(i) - 1) * patches .^ 2) + 1: (samples_to_display(i) * patches .^ 2)); 
    result_patches = result_yuv(:, :, :, ((samples_to_display(i) - 1) * patches .^ 2) + 1: (samples_to_display(i) * patches.^ 2));
    original_image = []; 
    result_image = []; 
    size(original_patches)
    size(result_patches)
    for j = 1 : patches
        original_col_patches = [];
        result_col_patches = [];
        for k = 1 : patches
            (j - 1) * patches + k
            original_col_patches = [original_col_patches; original_patches(:, :, :, (j - 1) * patches + k)];
            result_col_patches = [result_col_patches; result_patches(:, :, :, (j - 1) * patches + k)];
        end
        original_image = [original_image original_col_patches];
        result_image = [result_image result_col_patches];
    end
    original_rgb_image = yuv2rgb(original_image, 0);
    result_rgb_image = yuv2rgb(result_image, 0);
    figure();
    title('original');
    subplot(1, 2, 1);
    imshow(original_rgb_image);
    title('colorization');
    subplot(1, 2, 2);
    imshow(result_rgb_image);
end
