% Linear regression using least square gradient descent

clear all
load('PuntaCana_classification.mat');

% Constants
alpha = 0.003;

XTr = X_train;
yTr = y_train;
XTe = X_test;

% Dummy encode categorical variables
binVars = [7 9 32];
catVars = [15 26];
XTr = cleanData(XTr, binVars, catVars);
XTe = cleanData(XTe, binVars, catVars);

yTr(yTr == -1) = 0;

beta = logisticRegression(yTr, XTr, alpha);

% Predict error
[zol, ll, ~, ppp] = LogisticRegressionCost(yTr, XTr, beta);

fprintf('0-1 Loss: %f\n', zol);
fprintf('Log Loss: %f\n', ll);

% Predict test data
[~, ~, yTe] = LogisticRegressionCost(zeros(size(XTe, 1), 1), XTe, beta);

