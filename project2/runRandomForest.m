clearvars;

load train/train.mat;

K = 2;

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForest(xTrain, yTrain, xTest, yTest, 100, 74));

errors = crossval(fun, train.X_hog, train.y, 'kfold', K);
