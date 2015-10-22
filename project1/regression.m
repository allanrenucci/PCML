clear all

load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

X = normalize(X);

alpha = 0.1;
lambda = 1;

betaLS = leastSquares(y, tX);
betaGD = leastSquaresGD(y, tX, alpha);
betaRR = ridgeRegression(y, phi(X, 4), lambda);