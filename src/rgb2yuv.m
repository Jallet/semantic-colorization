function YUV = rgb2yuv(RGB, plot_flag)
RGB = im2double(RGB);
R = RGB(:, :, 1);
G = RGB(:, :, 2);
B = RGB(:, :, 3);

%Convert
convert_matrix = [0.299, 0.587, 0.114, 0;
                  -0.168736, -0.331264, 0.5, 128;
                  0.5, -0.418688, -0.81312, 128];
%Y = 0.299 * R + 0.587 * G + 0.114 * B;
%U = -0.168736 * R - 0.331264 * G + 0.5 * B;
%V = 0.5 * R - 0.418688 * G - 0.081312 * B;

Y = 0.299*R + 0.587*G + 0.114*B;
U = -0.147*R - 0.289*G + 0.436*B;
V = 0.615*R - 0.515*G - 0.100*B;

if (plot_flag == 1)
    figure();
    title('rgb2yuv');
    subplot(1, 3, 1);
    imshow(Y);
    title('Y');
    subplot(1, 3, 2);
    imshow(U);
    title('U');
    subplot(1, 3, 3);
    imshow(V);
    title('V');
end
YUV = cat(3, Y, U, V);
end
