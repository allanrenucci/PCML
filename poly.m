function Xpoly = poly(X,degree)
% build matrix Phi for polynomial regression of a given degree
    for l = 1:length(X)
        for k = 1:degree
            Xpoly(l, k) = sum(X(l, :).^k);
        end
    end
end

