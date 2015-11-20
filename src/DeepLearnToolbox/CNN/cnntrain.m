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
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y, size(x, 1), size(x, 2));
            opts.alpha = 1.0/(opts.c + opts.count);
            opts.count = opts.count + 1;
            net = cnnapplygrads(net, opts);
            disp(['Epoch ' num2str(i) '/' num2str(opts.numepochs) ', Batch: ' num2str(l) '/' num2str(numbatches), ', Loss: ' num2str(net.L) ', Gradient: ' num2str(net.gradient)]);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc
    end
    
end
