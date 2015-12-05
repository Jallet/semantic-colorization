function net = cnnff(net, x, opts)
    n = numel(net.layers);
    %net.layers{1}.a{1} = x;
    net.layers{1}.a = reshape(x, size(x, 1), size(x, 2), 1, size(x, 3));
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            net.layers{l}.a = zeros(net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), net.layers{l}.outputmaps, size(x, 3));
            a = net.layers{l - 1}.a;
            la = zeros(size(net.layers{l}.a));
            k = net.layers{l}.k;
            b = net.layers{l}.b;
            activation_type = opts.activation_type;
        
            parfor j = 1 : net.layers{l}.outputmaps   %  for each output map
                
                %  create temp output map
                %size of each output map

                %%%commented by Curio J
                %z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                %for i = 1 : inputmaps   %  for each input map
                %    %  convolve with corresponding kernel and add to temp output map
                %    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                %end
                %%%commented by Curio J
                
                %%%Add by Curio J 
                z = convn(a, k(:, :, :, j), 'valid');
                z = squeeze(z);
                %%%Add by Curio J 
                
                %  add bias, pass through nonlinearity
                %%%commented by Curio J
                %net.layers{l}.a{j} = activate(z + net.layers{l}.b{j}, opts.activation_type);
                %%%commented by Curio J
                
                %%%Add by Curio J 
                %tmp = z + net.layers{l}.b(j);
                la(:, :, j, :) = activate(z + b(j), activation_type);
                
                %%%Add by Curio J 
            end
           
            net.layers{l}.a = la;
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            scale = net.layers{l}.scale;
            net.layers{l}.a = zeros(net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), net.layers{l}.outputmaps, size(x, 3));
            a = net.layers{l - 1}.a;
            la = zeros(size(net.layers{l}.a));
            parfor j = 1 : inputmaps
                z = convn(a(:, :, j, :), ones(scale) / (scale ^ 2), 'valid');   %  !! replace with variable
                z = squeeze(z);
                la(:, :, j, :) = z(1 : scale : end, 1 : scale : end, :);
            end
            net.layers{l}.a = la;
            %z = convn(net.layers{l - 1}.a, ones(net.layers{l}.scale / (net.layers{l}.scale ^ 2)), 'valid'); 

        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    %%%commented by Curio J
    %for j = 1 : numel(net.layers{n}.a)
    %    sa = size(net.layers{n}.a{j});
    %    net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    %end
    %%%commented by Curio J
    
    %%%Add by Curio J
    feature_size = numel(net.layers{n}.a) / size(x, 3);
    net.fv = reshape(net.layers{n}.a, feature_size, size(x, 3));
    %%%Add by Curio J
    %  feedforward into output perceptrons
    %net.o = activate(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)), opts.activation_type);
    %net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    %net.o = softmax(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    net.o = softmax(bsxfun(@plus, net.ffW * net.fv, net.ffb));
end
