function L = computeCost(y, tX, beta)
%COMPUTECOST Compute MSE
%   Detailed explanation goes here
N = length(y);
e = y - tX * beta;
L = e' * e / (2 * N);

end

