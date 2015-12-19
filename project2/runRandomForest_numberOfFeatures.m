clearvars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = train.y;
clear train;

K = 10;

numFeatures = logspace(1, 3, 30);
numTrees = 60;

options = statset('UseParallel', false);

for i = 1:size(numFeatures, 2)
    nbFeatures = numFeatures(i);
    fprintf('Run with %d features...\n', nbFeatures);
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (randomForest(xTrain, yTrain, xTest, yTest, numTrees, nbFeatures, coeff));
    errs = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errs(:, 1)';
    berErrsTrain(i, :) = errs(:, 2)';
    predErrsTest(i, :) = errs(:, 3)';
    berErrsTest(i, :) = errs(:, 4)';
end

save('out/3_errors_per_nb_features.mat', 'numFeatures', 'predErrsTrain', 'berErrsTrain', 'predErrsTest', 'berErrsTest');

figure;
boxplot(predErrsTrain', numTrees, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(predErrsTest', numTrees, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

