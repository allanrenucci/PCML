clearvars;

load train/train.mat;

K = 2;

X = [train.X_hog, train.X_cnn];
%X = normalize(X, mean(X), 1);
X = zscore(X);
coeff = reduceDimension(X, 95);

fun = @(xTrain, yTrain, xTest, yTest) ...
    (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff));

errors = crossval(fun, X, train.y, 'kfold', K);
