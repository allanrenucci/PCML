clearvars;

load 'train/train.mat';
load 'out/4_concatenated_coeff_explained.mat';

y = train.y;
X = [train.X_hog train.X_cnn];
X = X * coeff;
clear train;

K = 10;
maxNeurons = size(X, 2);
numHiddenNeurons = unique(max(4, floor(logspace(0, log10(maxNeurons)))));

options = statset('UseParallel', true);

for i = 1:size(numHiddenNeurons, 2)
    
    numNeurons = ceil(numHiddenNeurons(i));
    fprintf('Run with %d hidden neurons [%d/%d]...\n', numNeurons, i, size(numHiddenNeurons, 2));
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff, numNeurons));
    errs = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errs(:, 1)';
    berErrsTrain(i, :) = errs(:, 2)';
    predErrsTest(i, :) = errs(:, 3)';
    berErrsTest(i, :) = errs(:, 4)';
end

save('out/4_neuralNetwork_errors_per_num_hidden_neurons.mat', 'predErrsTrain', 'berErrsTrain', 'predErrsTest', 'berErrsTest', 'numHiddenNeurons');

figure;
boxplot(predErrsTrain', variances, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(predErrsTest', variances, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

