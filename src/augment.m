function [aug_x, aug_y] = augment(x, y, patch)
    aug_x = zeros(size(x, 1) + patch.row_size - 1, size(x, 2) + patch.col_size - 1, size(x, 3));
    aug_y = zeros(size(x, 1) + patch.row_size - 1, size(x, 2) + patch.col_size - 1, size(x, 3));
    aug_x((patch.row_size + 1) / 2 : (size(aug_x, 1) - (patch.row_size - 1) / 2), (patch.col_size + 1) / 2 : (size(aug_x, 1) - (patch.col_size - 1) / 2), :) = x;
    y = reshape(y, size(x, 1), size(x, 2), size(y, 2));
    aug_y((patch.row_size + 1) / 2 : (size(aug_y, 1) - (patch.row_size - 1) / 2), (patch.col_size + 1) / 2 : (size(aug_y, 1) - (patch.col_size - 1) / 2), :) = y;
    aug_y = reshape(aug_y, size(aug_x, 1) * size(aug_x, 2), size(y, 3));
end
