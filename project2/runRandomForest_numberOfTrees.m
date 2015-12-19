clearvars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = train.y;
clear train;

K = 10;

numFeatures = sqrt(size(X, 2));
trees = 10:10:100;

options = statset('UseParallel', false);

for i = 1:size(trees, 2)
    nbTrees = trees(i);
    fprintf('Run with %d trees...\n', nbTrees);
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (randomForest(xTrain, yTrain, xTest, yTest, nbTrees, numFeatures, coeff));
    errs = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errs(:, 1)';
    berErrsTrain(i, :) = errs(:, 2)';
    predErrsTest(i, :) = errs(:, 3)';
    berErrsTest(i, :) = errs(:, 4)';
end

save('out/3_errors_per_nb_trees.mat', 'predErrsTrain', 'berErrsTrain', 'predErrsTest', 'berErrsTest');

figure;
boxplot(predErrsTrain', trees, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(predErrsTest', trees, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

