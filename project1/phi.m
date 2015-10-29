function phi = phi(X, degree)
% build matrix Phi for polynomial regression of a given degree
    phi = ones(size(X, 1), 1);
    for k = 1:degree
        phi = [phi X.^k];
    end
    
end

