function RGB = yuv2rgb(YUV, plot_flag)
Y = YUV(:, :, 1);
U = YUV(:, :, 2);
V = YUV(:, :, 3);

%Convert
%R = 2.0260 * Y + 1.8181 * U + 1.4020 * V;
%G = 0.4774 * Y - 1.2702 * U - 0.7141 * V;
%B = 1.0000 * Y + 1.7720 * U;

R = Y + 1.14*V;
G = Y - 0.39*U - 0.58*V;
B= Y + 2.03*U;

if (plot_flag == 1)
    figure();
    title('yuv2rgb')
    subplot(1, 3, 1);
    imshow(R);
    title('R');
    subplot(1, 3, 2);
    imshow(G);
    title('G');
    subplot(1, 3, 3);
    imshow(B);
    title('B');
end
RGB = cat(3, R, G, B);
RGB = uint8(RGB * 255);
end
