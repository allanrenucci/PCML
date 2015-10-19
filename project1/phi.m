function phi = phi(X,degree)
% build matrix Phi for polynomial regression of a given degree
    for k = 1:degree
        Xpoly(:, k) = sum(X.^k, 2);
    end
    
    phi = [ones(size(X, 1), 1) Xpoly];
end

