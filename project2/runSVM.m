clearvars;

load('train/train.mat');

K = 4;

X = train.X_cnn;

fun = @(xTrain, yTrain, xTest, yTest) ...
    (SVM(xTrain, yTrain, xTest, yTest));

options = statset('UseParallel', true);

errors = crossval(fun, X, train.y, 'kfold', K, 'Options', options);
