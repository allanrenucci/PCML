clear all

load('PuntaCana_classification.mat');
%load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

tX = [ones(length(y), 1) X];
%betaLS = leastSquares(y, tX);

% Normalize
meanX = mean(X);
X = X - ones(size(X)) * diag(meanX);
stdX = std(X);
X = X ./ (ones(size(X)) * diag(stdX));

tX = [ones(length(y), 1) X];
betaLS = leastSquares(y, tX);
alpha = 0.1;
betaGD = leastSquaresGD(y, tX, alpha);