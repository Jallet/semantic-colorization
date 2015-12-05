function net = cnnbp(net, y, opts)
    local_color_consistent_lambda = 0.5;
    n = numel(net.layers);
    %   error
    %net.e = net.o - y;
    tmp  = net.o;
    index = (1 : size(y, 2));
    class = accumarray([y' index'], 1, [opts.classes size(y, 2)]);
    %net.e = net.e - class;
    tmp = log(tmp);
    tmp = class .* tmp;
    net.L = -1 * sum(tmp(:)) / size(tmp, 2);
    %  loss function
    
    %square loss, commented by Curio
    %net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
    %%%

    %local color consistent
    %predict_color = reshape(net.o, [height, width, 2, size(y, 2)]);
    %color_consistent_weight = [0.1, 0.15, 0.1;
    %                           0.15, 0, 0.15;
    %                           0.1, 0.15, 0.1];
    %for i = 1 : size(y, 2)
    %    predict_weighted_color_u = conv2(predict_color(:, :, 1, i), color_consistent_weight, 'same');
    %    predict_weighted_color_v = conv2(predict_color(:, :, 2, i), color_consistent_weight, 'same');
    %    diff_u = abs(predict_weighted_color_u - predict_color(:, :, 1, i));
    %    diff_v = abs(predict_weighted_color_v - predict_color(:, :, 2, i));
    %     
    %    diff_sum = diff_u .^ 2 + diff_v .^ 2;
    
    
    %end
    %%  backprop deltas
    %net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    %net.fvd = (net.ffW' * net.od);              %  feature vector delta
    %if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
    %    net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    %end
    %tic
    %net.od = net.e .* derivate(net.o, opts.activation_type);   %  output delta

    %%%commented by Curio
    %net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    %%%
    %%%Add by Curio
    net.od = net.o - class;
    %%%
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* derivate(net.fv, opts.activation_type);
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a);
    %sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    %disp('1');
    %tic
    %%%commented by Curio J
    %for j = 1 : numel(net.layers{n}.a)
    %    j
    %    net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    %end
    %%%commented by Curio J
    %%%Add by Curio J
    net.layers{n}.d = reshape(net.fvd, size(net.layers{n}.a));
    %%%Add by Curio J
    %toc

    %disp('2');
    %tic
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            %%%commented by Curio J
            %for j = 1 : numel(net.layers{l}.a)
            %    net.layers{l}.d{j} = derivate(net.layers{l}.a{j}, opts.activation_type) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            %end
            %%%commented by Curio J
            net.layers{l}.d = derivate(net.layers{l}.a, opts.activation_type) .* (expand(net.layers{l + 1}.d, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1 1]) / net.layers{l + 1}.scale ^ 2);
        elseif strcmp(net.layers{l}.type, 's')
            net.layers{l}.d = zeros(size(net.layers{l}.a));
            k = net.layers{l + 1}.k;
            d = net.layers{l + 1}.d;
            ld = zeros(size(net.layers{l}.a));
            
            
            parfor i = 1 : size(net.layers{l}.a, 3)
                %for j = 1 : numel(net.layers{l + 1}.a)
                %     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                %end
                %net.layers{l}.d{i} = z;
                
                filter = squeeze(flipdim(flipdim(k(:, :, size(net.layers{l}.a, 3) - i + 1, :), 1), 2));
          
                filter = flipdim(filter, 3);
                filter = squeeze(filter);
                %filter = flipdim(filter, 3);
                %filter = flipdim(rot180(net.layers{l + 1}.k(:, :, i, :)), 3);
                
                %filter = squeeze(filter);
                
                aug_d = zeros(size(d, 1) + (size(filter, 1) - 1) * 2, size(d, 2) + (size(filter, 2) - 1) * 2, size(d, 3), size(d, 4));
               
                aug_d(size(filter, 1) : size(aug_d, 1) - size(filter, 1) + 1, size(filter, 2) : size(aug_d, 2) - size(filter, 2) + 1, :, :) = d;
                %filter = flipdim(filter, 3);
                z = convn(aug_d, filter, 'valid');
                
                ld(:, :, i, :) = z;
             
            end
            net.layers{l}.d = ld;
            
        end
    end
    %toc

    %toc

    %%  calc gradients
    %disp('bp grad');
    %tic
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            %net.layers{l}.dk = zeros(size(net.layers{l}.k));
            for j = 1 : size(net.layers{l}.a, 3)
                %la = net.layers{l - 1}.a;
                %tmp = convn(net.layers{l - 1}.a, net.layers{l}.d(:, :, j, :), 'valid');
                %net.layers{l}.dk(:, :, :, j) = convn(net.layers{l - 1}.a, net.layers{l}.d(:, :, j, :), 'valid') / size(net.layers{l}.d, 4);
                for i = 1 : size(net.layers{l - 1}.a, 3)    
                    %tmp = convn(flipall(squeeze(net.layers{l - 1}.a(:, :, i, :))), squeeze(net.layers{l}.d(:, :, j, :)), 'valid') / size(net.layers{l}.d, 4);
                    net.layers{l}.dk{i}{j} = convn(flipall(squeeze(net.layers{l - 1}.a(:, :, size(net.layers{l - 1}.a, 3) - i + 1, :))), squeeze(net.layers{l}.d(:, :, j, :)), 'valid') / size(net.layers{l}.d, 4);
                end
                d = net.layers{l}.d(:, :, j, :);
                net.layers{l}.db{j} = sum(d(:)) / size(net.layers{l}.d, 4);
            end
            %for j = 1 : numel(net.layers{l}.a)
            %    for i = 1 : numel(net.layers{l - 1}.a)
            %        net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
            %    end
            %    net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            %end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);
    %toc
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
