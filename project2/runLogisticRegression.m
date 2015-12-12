clearvars;

load train/train.mat;

K = 2;

X = train.X_hog;
%sigma = std(X); Fuck up the result ?
sigma = 1;
[coeff, ~, mu] = reduceDimension(normalize(X, 0, sigma), 95);
X = normalize(X, mu, sigma);

fun = @(xTrain, yTrain, xTest, yTest) ...
    (logisticRegression(xTrain, yTrain, xTest, yTest, coeff));

errors = crossval(fun, X, train.y, 'kfold', K);
