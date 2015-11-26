function [er, original_yuv, result_yuv] = cnntest(net, full_x, x, y, opts)
    %  feedforward
    disp('Feed forwarding...');
    tic
    net = cnnff(net, x, opts);
    toc
    %result_yuv = zeros(size(x, 1) - opts.row_size + 1, size(x, 2) - opts.col_size + 1, 3);
    %original_yuv = zeros(size(x, 1) - opts.row_size + 1, size(x, 2) - opts.col_size + 1, 3);
    result_yuv = zeros([size(full_x) 3]); 
    original_yuv = zeros([size(full_x) 3]); 
    %result_yuv(:, :, 1, :) = x((opts.row_size + 1) / 2 : size(x, 1) - (opts.row_size - 1) / 2, (opts.col_size + 1) / 2 : size(x, 2) - (opts.col_size - 1) / 2);
    result_yuv(:, :, 1) = full_x;    
    original_yuv(:, :, 1) = full_x;    
    %original_yuv(:, :, 1, :) = x((opts.row_size + 1) / 2 : size(x, 1) - (opts.row_size - 1) / 2, (opts.col_size + 1) / 2 : size(x, 2) - (opts.col_size - 1) / 2);
    original_yuv(:, :, 2) = reshape(y(1, :), size(original_yuv, 2), size(original_yuv, 1));
    original_yuv(:, :, 2) = original_yuv(:, :, 2)';
    original_yuv(:, :, 3) = reshape(y(2, :), size(original_yuv, 2), size(original_yuv, 1));
    original_yuv(:, :, 3) = original_yuv(:, :, 3)';
    
    result_yuv(:, :, 2) = reshape(net.o(1, :), size(result_yuv, 2), size(result_yuv, 1));
    result_yuv(:, :, 2) = result_yuv(:, :, 2)';
    result_yuv(:, :, 3) = reshape(net.o(2, :), size(result_yuv, 2), size(result_yuv, 1));
    result_yuv(:, :, 3) = result_yuv(:, :, 3)';
    
    er = ((result_yuv(:, :, 2) - original_yuv(:, :, 2)) .^ 2 + (result_yuv(:, :, 3) - original_yuv(:, :, 3)) .^ 2) / (size(result_yuv, 1) * size(result_yuv, 2));
    er = sum(er(:));
end
