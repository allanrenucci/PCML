clearvars;

load 'train/train.mat';
load 'out/4_concatenated_coeff_explained.mat';

y = train.y;
X = [train.X_hog train.X_cnn];
X = X * coeff;
clear train;

K = 10;
seeds = randi(10000, 1, 20);

options = statset('UseParallel', false);

for i = 1:size(seeds, 2)
    
    seed = seeds(i);
    fprintf('Run with seed %d [%d/%d]...\n', seed, i, size(seeds, 2));
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (neuralNetwork(xTrain, yTrain, xTest, yTest, seed));
    errs = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errs(:, 1)';
    berErrsTrain(i, :) = errs(:, 2)';
    predErrsTest(i, :) = errs(:, 3)';
    berErrsTest(i, :) = errs(:, 4)';
end

save('out/4_neuralNetwork_errors_per_seeds.mat', 'predErrsTrain', 'berErrsTrain', 'predErrsTest', 'berErrsTest', 'seeds');

figure;
boxplot(predErrsTrain', seeds, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(predErrsTest', seeds, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

