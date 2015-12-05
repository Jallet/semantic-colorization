function X = relu(P)
    X = P .* (P > 0);
end
