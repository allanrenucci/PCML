clearvars;

load 'train/train.mat';
load 'concatenated_std1.mat';

K = 2;

numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForest(xTrain, yTrain, xTest, yTest, 200, numFeatures, coeff));

errors = crossval(fun, X, train.y, 'kfold', K);
