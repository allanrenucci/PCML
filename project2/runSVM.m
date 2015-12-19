clearvars;

load('train/train.mat');

K = 5;

X = [train.X_cnn train.X_hog];

fun = @(xTrain, yTrain, xTest, yTest) ...
    (SVM(xTrain, yTrain, xTest, yTest));

options = statset('UseParallel', false);

errors = crossval(fun, X, train.y, 'kfold', K, 'Options', options);
