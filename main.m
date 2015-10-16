clear all

load('PuntaCana_classification.mat');
%load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

tX = [ones(length(y), 1) X];
%betaLS = leastSquares(y, tX);

X = normalize(X);

tX = [ones(length(y), 1) X];
alpha = 0.1;
lambda = 1;
betaLS = leastSquares(y, tX);
betaGD = leastSquaresGD(y, tX, alpha);
ridge = ridgeRegression(y, tX, lambda);