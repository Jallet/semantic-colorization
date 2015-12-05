function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];

    for i = 1 : opts.numepochs
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            tic
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
           
            net = cnnff(net, batch_x, opts);
            
            [~, output_label] = max(net.o);
            bad = (batch_y ~= output_label);
            if ~isfield(net, 'er')
                net.er(1) = double(sum(bad(:)) / size(batch_y, 2));
            else
                net.er(end + 1) = double(sum(bad(:)) / size(batch_y, 2));
            end
            
            net = cnnbp(net, batch_y, opts);
            
            opts.alpha = 1.0/(opts.c + opts.count);
            opts.count = opts.count + 1;
            
            net = cnnapplygrads(net, opts);
            
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            if ~isfield(net, 'loss')
                net.loss(1) = net.L;
            else
                net.loss(end + 1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            disp(['Epoch ' num2str(i) '/' num2str(opts.numepochs) ', Batch: ' num2str(l) '/' num2str(numbatches), ', net.L: ' num2str(net.L) ', net.rL: ' num2str(net.rL(end)) ', Gradient: ' num2str(net.gradient) ', error: ' num2str(net.er(end))]);
            toc
        end
        toc
    end
    
end
