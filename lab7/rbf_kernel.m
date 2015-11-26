function [ K ] = rbf_kernel( X1, X2, gamma)
    %RBF_KERNEL Compute the radial basis function kernel matrix.
    dists = pdist2(X1, X2);    
    K = exp(-gamma * (dists.^2));
end

