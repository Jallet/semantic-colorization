function X = activate(P, type)
    switch(type)
    case 'sigmoid'
        X = sigm(P);
    case 'relu'
        X = relu(P);
    end
end
