function beta = leastSquares(y, tX)
%LEASTSQUARE Summary of this function goes here
%   Detailed explanation goes here

%beta = inv(tX' * tX) * tX' * y;
beta = (tX' * tX) \ (tX' * y);
end

