function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    result_yuv = zeros(size(x, 1), size(x, 2), 3, size(x, 3));
    original_yuv = zeros(size(x, 1), size(x, 2), 3, size(x, 3));
    result_yuv(:, :, 1, :) = x;
    original_yuv(:, :, 1, :) = x;
    result_uv = reshape(net.o, size(x, 1), size(x, 2), 2, size(x, 3));
    result_yuv(:, :, 2 : 3, :) = result_uv;
    original_uv = reshape(y, size(x, 1), size(x, 2), 2, size(x, 3));
    original_yuv(:, :, 2 : 3, :) = original_uv;
    save('/home/jiangliang/code/semantic-colorization/result/result_yuv', 'result_yuv');
    save('/home/jiangliang/code/semantic-colorization/result/original_yuv', 'original_yuv');
     
    %samples_to_display = randperm(size(y, 2));
    %samples_to_display = samples_to_display(1 : 4);
    %for i = 1 : length(samples_to_display)
    %    original_image = original_yuv(:, :, :, samples_to_display(i)); 
    %    result_image = result_yuv(:, :, :, samples_to_display(i)); 
    %    original_rgb_image = yuv2rgb(original_image, 0);
    %    result_rgb_image = yuv2rgb(result_image, 0);
    %    figure();
    %    title('original');
    %    subplot(1, 2, 1);
    %    imshow(original_rgb_image);
    %    title('colorization');
    %    subplot(1, 2, 2);
    %    imshow(result_rgb_image);

    %end

    %[~, h] = max(net.o);
    %[~, a] = max(y);
    %bad = find(h ~= a);
    bad = 0;
    er = 0;
    %er = numel(bad) / size(y, 2);
end
