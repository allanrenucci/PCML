clearvars;

load train/train.mat;

K = 2;

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForest(xTrain, yTrain, xTest, yTest, 200, 74));

errors = crossval(fun, train.X_cnn, train.y, 'kfold', K);
