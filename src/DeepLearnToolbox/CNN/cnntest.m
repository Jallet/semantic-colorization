function [er, original_yuv, original_class_yuv, result_yuv] = cnntest(net, full_x, full_y, x, y, center, opts)
    %  feedforward
    disp('Feed forwarding...');
    tic
    net = cnnff(net, x, opts);
    toc
    %result_yuv = zeros(size(x, 1) - opts.row_size + 1, size(x, 2) - opts.col_size + 1, 3);
    %original_yuv = zeros(size(x, 1) - opts.row_size + 1, size(x, 2) - opts.col_size + 1, 3);
    result_yuv = zeros([size(full_x) 3]); 
    original_yuv = zeros([size(full_x) 3]); 
    original_class_yuv = zeros([size(full_x) 3]); 
    %result_yuv(:, :, 1, :) = x((opts.row_size + 1) / 2 : size(x, 1) - (opts.row_size - 1) / 2, (opts.col_size + 1) / 2 : size(x, 2) - (opts.col_size - 1) / 2);
    result_yuv(:, :, 1) = full_x;    
    original_yuv(:, :, 1) = full_x;    
    original_class_yuv(:, :, 1) = full_x;    
    %original_yuv(:, :, 1, :) = x((opts.row_size + 1) / 2 : size(x, 1) - (opts.row_size - 1) / 2, (opts.col_size + 1) / 2 : size(x, 2) - (opts.col_size - 1) / 2);
    
    %original_yuv(:, :, 2) = reshape(y(1, :), size(original_yuv, 2), size(original_yuv, 1));
    %original_yuv(:, :, 2) = original_yuv(:, :, 2)';
    %original_yuv(:, :, 3) = reshape(y(2, :), size(original_yuv, 2), size(original_yuv, 1));
    %original_yuv(:, :, 3) = original_yuv(:, :, 3)';
  
    original_yuv(:, :, 2 : 3) = reshape(full_y, size(original_yuv, 1), size(original_yuv, 2), 2);
    disp('size')
    original_center = center(y, :);
    original_class_yuv(:, :, 2) = reshape(original_center(:, 1), size(original_class_yuv, 2), size(original_class_yuv, 1))'; 
    original_class_yuv(:, :, 3) = reshape(original_center(:, 2), size(original_class_yuv, 2), size(original_class_yuv, 1))'; 
   
    [~, result_class] = max(net.o);
    result_center = center(result_class, :);
    size(result_center)
    
    result_yuv(:, :, 2) = reshape(result_center(:, 1), size(result_yuv, 2), size(result_yuv, 1))'; 
    result_yuv(:, :, 3) = reshape(result_center(:, 2), size(result_yuv, 2), size(result_yuv, 1))'; 
    
    %result_yuv(:, :, 2) = reshape(net.o(1, :), size(result_yuv, 2), size(result_yuv, 1));
    %result_yuv(:, :, 2) = result_yuv(:, :, 2)';
    %result_yuv(:, :, 3) = reshape(net.o(2, :), size(result_yuv, 2), size(result_yuv, 1));
    %result_yuv(:, :, 3) = result_yuv(:, :, 3)';
    %bad = 0;
    %er = 0;
    bad = y ~= result_class;
    er = sum(bad(:)) / size(y, 2); 
end
