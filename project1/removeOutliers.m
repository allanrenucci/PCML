function [ X, y ] = removeOutliers(X, y)

mu = mean(X);
sigma = std(X);

[n, ~] = size(X);
% Create a matrix of mean values by
% replicating the mu vector for n rows
MeanMat = repmat(mu, n, 1);
% Create a matrix of standard deviation values by
% replicating the sigma vector for n rows
SigmaMat = repmat(sigma, n, 1);
% Create a matrix of zeros and ones, where ones indicate
% the location of outliers
outliers = abs(X - MeanMat) > 3 * SigmaMat;

X(any(outliers, 2), :) = [];
y(any(outliers, 2), :) = [];

end

