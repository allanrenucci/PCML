clear all

load('PuntaCana_classification.mat');
%load('PuntaCana_regression.mat');

tX = [ones(length(y_train), 1) X_train];
betaLS = leastSquares(y_train, tX);