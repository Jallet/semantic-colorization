function [x_patches, y_patches] = split_data(x, y, row_patches, col_patches)
    
    for i = 1 : size(y, 2)
        sample_x = x(:, :, i);
        patches = mat2cell(sample_x, [size(sample_x, 1) / row_patches * ones(1, row_patches)], [size(sample_x, 2) / col_patches * ones(1, col_patches)]);
        num_patches = row_patches * col_patches;
        for j = 1 : num_patches
            x_patches(:, :, (i - 1) * num_patches + j) = patches{j};
        end
    
        sample_y = y(:, i);
        sample_y = reshape(sample_y, [size(x, 1), size(x, 2), 2]);
        patches_u = mat2cell(sample_y(:, :, 1), size(sample_y(:, :, 1), 1) / row_patches * ones(1, row_patches), size(sample_y(:, :, 1), 2) / col_patches * ones(1, col_patches));
        patches_v = mat2cell(sample_y(:, :, 2), size(sample_y(:, :, 2), 1) / row_patches * ones(1, row_patches), size(sample_y(:, :, 2), 2) / col_patches * ones(1, col_patches));
            for j = 1 : num_patches
                u = patches_u{j};
                v = patches_v{j};
                y_patches(:, (i - 1) * num_patches + j) = [u(:); v(:)];
            end
    end

end
