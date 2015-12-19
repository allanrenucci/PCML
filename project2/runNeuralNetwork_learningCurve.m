clearvars;

load 'train/train.mat';
load 'out/1_concatenated_std1_var95.mat';

y = train.y;
clear train;

props = linspace(0.1, 0.9, 41);
numTrees = 60;
numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
        (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff));

learningCurve(fun, X, y, props);
