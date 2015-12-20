clearvars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = zeros(size(train.y));
y(train.y == 4) = 1;
y(train.y ~= 4) = 2;
clear train;

K = 10;

options = statset('UseParallel', false);

numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForestBinary(xTrain, yTrain, xTest, yTest, 100, numFeatures, coeff));

errors = crossval(fun, X, y, 'kfold', K, 'Options', options);
