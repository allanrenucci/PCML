clearvars;

load 'train/train.mat';
load 'concatenated_std1.mat';

K = 5;

numFeatures = sqrt(size(X, 2));
trees = 10:10:100;

for i = 1:size(trees, 2)
    nbTrees = trees(i);
    fprintf('Run with %d trees...\n', nbTrees);
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (randomForest(xTrain, yTrain, xTest, yTest, nbTrees, numFeatures, coeff));
    errs = crossval(fun, X, train.y, 'kfold', K);
    predErrs(i, :) = errs(:, 1)';
    berErrs(i, :) = errs(:, 2)';
end

save('out/errors_per_nb_trees.mat', 'predErrs', 'berErrs');

figure;
boxplot(predErrs', trees, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

