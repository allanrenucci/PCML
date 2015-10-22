clear all

load('PuntaCana_classification.mat');
%load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

y(y == -1) = 0;
X = normalize(X);

[XTr, yTr, XTe, yTe] = split(y, X, 0.5);

tXTr = [ones(length(yTr), 1) XTr];
alpha = 0.1;
lambda = 1;
betaLS = leastSquares(yTr, tXTr);
betaGD = leastSquaresGD(yTr, tXTr, alpha);
betaRR = ridgeRegression(yTr, phi(XTr, 4), lambda);
betaLR = logisticRegression(yTr, tXTr, alpha);
betaPLR = penLogisticRegression(yTr, tXTr, alpha, lambda);

