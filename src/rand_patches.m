function [x_patches, y_patches] = rand_patches(x, y, patch)
    y_matrix = reshape(y, size(x, 1), size(x, 2), 2, size(y, 2));
    rand_image_index = randint(patch.num, 1, [1, size(y, 2)]);
    rand_row = randint(patch.num, 1, [(patch.row_size + 1) / 2, size(x, 1) - (patch.row_size - 1) / 2]);  
    rand_col = randint(patch.num, 1, [(patch.col_size + 1) / 2, size(x, 2) - (patch.col_size - 1) / 2]);  
    x_patches = zeros(patch.row_size, patch.col_size, patch.num);
    y_patches = zeros(2, patch.num);
    for i = 1 : patch.num
        x_patches(:, :, i) = x(rand_row(i) - (patch.row_size - 1) / 2 : rand_row(i) + (patch.col_size - 1) / 2, rand_col(i) - (patch.col_size - 1) / 2 : rand_col(i) + (patch.col_size - 1) / 2);
        y_patches(:, i) = [y_matrix(rand_row(i), rand_col(i), 1); y_matrix(rand_row(i), rand_col(i), 2)];
    end

    %for i = 1 : size(y, 2)
    %    sample_x = x(:, :, i);
    %    patches = mat2cell(sample_x, [size(sample_x, 1) / row_patches * ones(1, row_patches)], [size(sample_x, 2) / col_patches * ones(1, col_patches)]);
    %    num_patches = row_patches * col_patches;
    %    for j = 1 : num_patches
    %        x_patches(:, :, (i - 1) * num_patches + j) = patches{j};
    %    end
    %
    %    sample_y = y(:, i);
    %    sample_y = reshape(sample_y, [size(x, 1), size(x, 2), 2]);
    %    patches_u = mat2cell(sample_y(:, :, 1), size(sample_y(:, :, 1), 1) / row_patches * ones(1, row_patches), size(sample_y(:, :, 1), 2) / col_patches * ones(1, col_patches));
    %    patches_v = mat2cell(sample_y(:, :, 2), size(sample_y(:, :, 2), 1) / row_patches * ones(1, row_patches), size(sample_y(:, :, 2), 2) / col_patches * ones(1, col_patches));
    %        for j = 1 : num_patches
    %            u = patches_u{j};
    %            v = patches_v{j};
    %            mini_u = u(size(u, 1) / 2 : size(u, 1) / 2 + 1, size(u, 2) / 2 : size(u, 2) / 2 + 1 );
    %            mini_v = v(size(v, 1) / 2 : size(v, 1) / 2 + 1, size(v, 2) / 2 : size(v, 2) / 2 + 1 );
    %            %y_patches(:, (i - 1) * num_patches + j) = [u(:); v(:)];
    %            y_patches(:, (i - 1) * num_patches + j) = [mini_u(:); mini_v(:)];
    %        end
    %end

end
