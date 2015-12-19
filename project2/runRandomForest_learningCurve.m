clearvars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = train.y;
clear train;

props = linspace(0.1, 0.9, 41);
numTrees = 60;
numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
        (randomForest(xTrain, yTrain, xTest, yTest, numTrees, numFeatures, coeff));

learningCurve(fun, X, y, props);
