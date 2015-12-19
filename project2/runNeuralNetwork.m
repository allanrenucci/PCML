clearvars;

load 'train/train.mat';
%load 'out/2_concatenated_std1_var100.mat';
load 'out/1_concatenated_std1_var95.mat';

y = train.y;
clear train;

K = 5;

options = statset('UseParallel', false);

fun = @(xTrain, yTrain, xTest, yTest) ...
    (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff));

errors = crossval(fun, X, y, 'kfold', K, 'Options', options);
