function g = computeGradient(y, tX, beta)
%COMPUTEGRADIE Summary of this function goes here
%   Detailed explanation goes here
N = length(y);
e = y - tX * beta;
g = -(tX' * e) / N;

end

