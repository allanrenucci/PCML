clearvars;

load train/train.mat;

K = 2;

fun = @(xTrain, yTrain, xTest, yTest) ...
    (neuralNetwork(xTrain, yTrain, xTest, yTest));

errors = crossval(fun, train.X_hog, train.y, 'kfold', K);
