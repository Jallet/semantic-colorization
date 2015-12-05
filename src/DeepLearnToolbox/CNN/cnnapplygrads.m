function net = cnnapplygrads(net, opts)
net.gradient = 0;
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            %for j = 1 : numel(net.layers{l}.a)
            %    for ii = 1 : numel(net.layers{l - 1}.a)
            %        net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
            %        gradient = net.layers{l}.dk{ii}{j};
            %        gradient = gradient .^ 2;
            %        net.gradient = net.gradient + sum(gradient(:));
            %    end
            %    net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            %    gradient = net.layers{l}.db{j};
            %    gradient = gradient .^ 2;
            %    net.gradient = net.gradient + sum(gradient(:));
            %end
            for j = 1 : size(net.layers{l}.a, 3)
                for ii = 1 : size(net.layers{l - 1}.a, 3)
                    net.layers{l}.k(:, :, ii, j) = net.layers{l}.k(:, :, ii, j) - opts.alpha * net.layers{l}.dk{ii}{j};
                    gradient = net.layers{l}.dk{ii}{j};
                    gradient = gradient .^ 2;
                    net.gradient = net.gradient + sum(gradient(:));
                    net.layers{l}.b(j) = net.layers{l}.b(j) - opts.alpha * net.layers{l}.db{j};
                    gradient = net.layers{l}.db{j};
                    gradient = gradient .^ 2;
                    net.gradient = net.gradient + sum(gradient(:));
                end
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;
    gradient = net.dffW;
    gradient = gradient .^ 2;
    net.gradient = net.gradient + sum(gradient(:));
    net.ffb = net.ffb - opts.alpha * net.dffb;
    gradient = net.dffb;
    gradient = gradient .^ 2;
    net.gradient = net.gradient + sum(gradient(:));
end
