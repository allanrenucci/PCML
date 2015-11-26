function [ K ] = linear_kernel( X1, X2 )
    %LINEAR_KERNEL Build a linear kernel.
    K = phi(X1, 1) * phi(X2, 1)';
end

