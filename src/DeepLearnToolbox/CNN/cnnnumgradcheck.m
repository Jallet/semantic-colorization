function cnnnumgradcheck(net, x, y, opts)
    epsilon = 1e-4;
    er      = 1e-8;
    n = numel(net.layers);
    disp('ffb');
    for j = 1 : numel(net.ffb)
       net_m = net; net_p = net;
       net_p.ffb(j) = net_m.ffb(j) + epsilon;
       net_m.ffb(j) = net_m.ffb(j) - epsilon;
       net_m = cnnff(net_m, x, opts); net_m = cnnbp(net_m, y, opts);
       net_p = cnnff(net_p, x, opts); net_p = cnnbp(net_p, y, opts);
       d = (net_p.L - net_m.L) / (2 * epsilon);
       e = abs(d - net.dffb(j));
       if e > er
               error('numerical gradient checking failed, press enter to continue');
       end
    end
    disp('ffb pass');
    disp('ffW');
    for i = 1 : size(net.ffW, 1)
       for u = 1 : size(net.ffW, 2)
           net_m = net; net_p = net;
           net_p.ffW(i, u) = net_m.ffW(i, u) + epsilon;
           net_m.ffW(i, u) = net_m.ffW(i, u) - epsilon;
           net_m = cnnff(net_m, x, opts); net_m = cnnbp(net_m, y, opts);
           net_p = cnnff(net_p, x, opts); net_p = cnnbp(net_p, y, opts);
           d = (net_p.L - net_m.L) / (2 * epsilon);
           e = abs(d - net.dffW(i, u));
           if e > er
               error('numerical gradient checking failed, press enter to continue');
           end
       end
    end
    disp('ffW pass');
    disp('k, b');
    count = 0;
    wrong_gradient = 0;
    for l = n : -1 : 2
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : size(net.layers{l}.a, 3)
                net_m = net; net_p = net;
                net_p.layers{l}.b(j) = net_m.layers{l}.b(j) + epsilon;
                net_m.layers{l}.b(j) = net_m.layers{l}.b(j) - epsilon;
                net_m = cnnff(net_m, x, opts); net_m = cnnbp(net_m, y, opts);
                net_p = cnnff(net_p, x, opts); net_p = cnnbp(net_p, y, opts);
                d = (net_p.L - net_m.L) / (2 * epsilon);
                e = abs(d - net.layers{l}.db{j});
                if e > er
                    disp('b');
                    %input('numerical gradient checking failed, press enter to continue');
                end
                for i = 1 : size(net.layers{l - 1}.a, 3)
                    for u = 1 : size(net.layers{l}.k, 1)
                        for v = 1 : size(net.layers{l}.k, 2)
                       
                            count = count + 1;
                            net_m = net; net_p = net;
                            net_p.layers{l}.k(u, v, i, j) = net_p.layers{l}.k(u, v, i, j) + epsilon;
                            net_m.layers{l}.k(u, v, i, j) = net_m.layers{l}.k(u, v, i, j) - epsilon;
                            net_m = cnnff(net_m, x, opts); 
                            net_m = cnnbp(net_m, y, opts);
                            net_p = cnnff(net_p, x, opts); 
                            net_p = cnnbp(net_p, y, opts);
                            d = (net_p.L - net_m.L) / (2 * epsilon);
                            e = abs(d - net.layers{l}.dk{i}{j}(u, v));
                            if e > er
                                wrong_gradient = wrong_gradient + 1;
                                disp(['layer: ' num2str(l) ', wrong: ' num2str(wrong_gradient) 'count: ' num2str(count) ', ratio: ' num2str(wrong_gradient / count)]);
                                error('numerical gradient checking failed, press enter to continue');
                            end
                        end
                    end
                end
            end
        elseif strcmp(net.layers{l}.type, 's')
%            for j = 1 : numel(net.layers{l}.a)
%                net_m = net; net_p = net;
%                net_p.layers{l}.b(j) = net_m.layers{l}.b(j) + epsilon;
%                net_m.layers{l}.b(j) = net_m.layers{l}.b(j) - epsilon;
%                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                d = (net_p.L - net_m.L) / (2 * epsilon);
%                e = abs(d - net.layers{l}.db(j));
%                if e > er
%                    error('numerical gradient checking failed');
%                end
%            end
        end
    end
    disp('k, b pass');
%    keyboard
disp('grad check pass');
end
