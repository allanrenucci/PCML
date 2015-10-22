clear all

load('PuntaCana_classification.mat');
%load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

y(y == -1) = 0;
X = normalize(X);

tX = [ones(length(y), 1) X];
alpha = 0.1;
lambda = 1;
%betaLS = leastSquares(y, tX);
%betaGD = leastSquaresGD(y, tX, alpha);
%betaRR = ridgeRegression(y, phi(X, 4), lambda);
betaLR = logisticRegression(y, tX, alpha);
betaPLR = penLogisticRegression(y, tX, alpha, lambda);