clearvars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = train.y;
clear train;

K = 10;

options = statset('UseParallel', true);

numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForest(xTrain, yTrain, xTest, yTest, 100, numFeatures, coeff));

errors = crossval(fun, X, y, 'kfold', K, 'Options', options);
