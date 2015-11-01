% Linear regression using least square gradient descent

clear all
load('PuntaCana_regression.mat');

% Constants
K = 10;
% Model 1: feature 95 >= 0.5
m1Degree = 1;
m1Lambda = 106.673394;
% Model 2: feature 95 < 0.5
m2Degree = 3;
m2Lambda = 2.998844;

XTr = X_train;
yTr = y_train;
XTe = X_test;

% Dummy encode categorical variables
binVars = [39 48 49];
catVars = [11 34 40 42 50 67 72];
XTr(:, catVars) = XTr(:, catVars) + 1;
XTe(:, catVars) = XTe(:, catVars) + 1;
XTr = cleanData(XTr, binVars, catVars);
XTe = cleanData(XTe, binVars, catVars);

% Remove some features (2.7% reduction on RMSE: 663 -> 645)
toRemove = [1 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27 31 36 39 41 42 44 47 48 51 53 56 57 59 61 62 63 65 66 67 68 69 70 73 76 77 80 81 82 83 85 86];
XTr = removeCols(XTr, toRemove);
XTe = removeCols(XTe, toRemove);

N = size(XTr, 2);

% Feature used to choose between the two models (last one in X)
index = N;

% Model 1
varsTrM1 = XTr(:, N) >= 0.5;
XTrM1 = XTr(varsTrM1, :);
yTrM1 = yTr(varsTrM1);
varsTeM1 = XTe(:, N) >= 0.5;
XTeM1 = XTe(varsTeM1, :);

% Model 2
varsTrM2 = XTr(:, N) < 0.5;
XTrM2 = XTr(varsTrM2, :);
yTrM2 = yTr(varsTrM2);
varsTeM2 = XTe(:, N) < 0.5;
XTeM2 = XTe(varsTeM2, :);

% form tX
tXTrM1 = phi(XTrM1, m1Degree);
tXTeM1 = phi(XTeM1, m1Degree);
tXTrM2 = phi(XTrM2, m2Degree);
tXTeM2 = phi(XTeM2, m2Degree);

betam1 = ridgeRegression(yTrM1, tXTrM1, m1Lambda);
betam2 = ridgeRegression(yTrM2, tXTrM2, m2Lambda);

% Predict test data
yTe = zeros(size(X_test, 2), 1);
yTe(varsTeM1) = tXTeM1 * betam1;
yTe(varsTeM2) = tXTeM2 * betam2;
