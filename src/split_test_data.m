function [x_patches, y_patches] = split_test_data(x, y, patch)

reshape_y = reshape(y, size(x, 1), size(x, 2), 2, size(x, 3));
num_rows = size(x, 1) - patch.row_size + 1;
num_cols = size(x, 2) - patch.col_size + 1;
x_patches = zeros(patch.row_size, patch.col_size, num_rows * num_cols, size(y, 2));
y_patches = zeros(2, num_rows * num_cols, size(y, 2));
num_cores = 16;
for l = 1 : size(y, 2)
    for i = (patch.row_size + 1) / 2 : size(x, 1) - (patch.row_size - 1) / 2
        row_x = zeros(patch.row_size, patch.col_size, num_cols);
        row_y = zeros(2, num_cols);
        for j = (patch.col_size + 1) / 2 : size(x, 2) - (patch.col_size - 1) / 2
            temp_x = zeros(patch.row_size, patch.col_size);
            temp_x = x(i - (patch.row_size - 1) / 2 : i + (patch.row_size - 1) / 2, j - (patch.col_size - 1) / 2 : j + (patch.col_size - 1) / 2, l);    
            row_x(:, :, j - (patch.col_size - 1) / 2) = temp_x;
            temp_y = zeros(2, 1);
            temp_y = [reshape_y(i, j, 1, l); reshape_y(i, j, 2, l)];
            row_y(:, j - (patch.col_size - 1) / 2) = temp_y;
        end
        x_patches(:, :, (i - (patch.col_size - 1) / 2 - 1) * num_rows + 1 : (i - (patch.col_size - 1) / 2) * num_rows, l) = row_x;
        y_patches(:, (i - (patch.col_size - 1) / 2 - 1) * num_rows + 1: (i - (patch.col_size - 1) / 2) * num_rows, l) = row_y;
    end
end
end
