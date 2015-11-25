function P = derivate(X, type)
    switch(type)
    case 'sigmoid'
        P = X .* (1 - X);
    case 'relu'
        P = X > 0;
    end
end
