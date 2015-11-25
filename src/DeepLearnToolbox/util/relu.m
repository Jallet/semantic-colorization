function X = sigm(P)
    X = P .* (P > 0);
end
