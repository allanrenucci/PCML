clear all

load('PuntaCana_classification.mat');

X = X_train;
y = y_train;

y(y == -1) = 0;
X = normalize(X);

alpha = 0.1;
lambda = 1;

betaLR = logisticRegression(y, tX, alpha);
betaPLR = penLogisticRegression(y, tX, alpha, lambda);