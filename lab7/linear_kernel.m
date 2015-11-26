function [ K ] = linear_kernel( X1, X2 )
    %LINEAR_KERNEL Build a linear kernel.
    K = phi(X1) * phi(X2)';
end

