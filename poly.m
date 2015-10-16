function Xpoly = poly(X,degree)
% build matrix Phi for polynomial regression of a given degree
    for k = 1:degree
        Xpoly(:,k) = sum(X.^k);
    end
end

