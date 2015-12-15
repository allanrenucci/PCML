clearvars;

load train/train.mat;

K = 2;

X = train.X_hog;
X = normalize(X, mean(X), std(X));
[coeff, ~, mu] = reduceDimension(X, 95);

fun = @(xTrain, yTrain, xTest, yTest) ...
    (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff));

errors = crossval(fun, X, train.y, 'kfold', K);
