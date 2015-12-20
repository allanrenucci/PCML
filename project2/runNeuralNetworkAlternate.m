clearvars;

load 'train/train.mat';
%load 'out/2_concatenated_std1_var100.mat';
load 'out/1_concatenated_std1_var95.mat';

y = train.y;
X = [train.X_hog train.X_cnn];
X = X * coeff;
clear train;

K = 10;

options = statset('UseParallel', false);

thresholds = linspace(0.5, 1, 21);

for i = 1:size(thresholds, 2)
    threshold = thresholds(i);
    fprintf('Run with confidence threshold = %f...\n', threshold);
    
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (neuralNetworkAlternate(xTrain, yTrain, xTest, yTest, 10, threshold));
    errors = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errors(:, 1)';
    berErrsTrain(i, :) = errors(:, 2)';
    predErrsTest(i, :) = errors(:, 3)';
    berErrsTest(i, :) = errors(:, 4)';
end
