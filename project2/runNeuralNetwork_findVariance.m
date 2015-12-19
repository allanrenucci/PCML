clearvars;

load 'train/train.mat';
load 'out/4_concatenated_coeff_explained.mat';

y = train.y;
X = [train.X_hog train.X_cnn];
clear train;

K = 10;

variances = linspace(50, 95, 10);

options = statset('UseParallel', false);

for i = 1:size(variances, 2)
    
    % Percentage of the variance retained
    retained = variances(i);
    sum_explained = 0;
    selected = 1;

    % Select columns until we reach the desired threshold
    while sum_explained < retained && selected <= length(explained)
        sum_explained = sum_explained + explained(selected);
        selected = selected + 1;
    end

    % Return the matrice with only the selected columns
    coeff_selected = coeff(:, 1:selected);
    
    fprintf('Run with variance = %.2f%%...\n', retained);
    fun = @(xTrain, yTrain, xTest, yTest) ...
        (neuralNetwork(xTrain, yTrain, xTest, yTest, coeff_selected));
    errs = crossval(fun, X, y, 'kfold', K, 'Options', options);
    predErrsTrain(i, :) = errs(:, 1)';
    berErrsTrain(i, :) = errs(:, 2)';
    predErrsTest(i, :) = errs(:, 3)';
    berErrsTest(i, :) = errs(:, 4)';
end

save('out/4_neuralNetwork_errors_per_variances.mat', 'predErrsTrain', 'berErrsTrain', 'predErrsTest', 'berErrsTest', 'variances');

figure;
boxplot(predErrsTrain', variances, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(predErrsTest', variances, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

