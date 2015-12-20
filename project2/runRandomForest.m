clear vars;

load 'train/train.mat';
load 'out/3_concatenated_non_normalized_var95.mat';

y = train.y;
clear train;

K = 10;

options = statset('UseParallel', true);

numFeatures = sqrt(size(X, 2));

fun = @(xTrain, yTrain, xTest, yTest) ...
    (randomForest(xTrain, yTrain, xTest, yTest, 60, numFeatures, coeff));

errors = crossval(fun, X, y, 'kfold', K, 'Options', options);

rng(0, 'twister');

finalBerErrsTest = zeros(20, 10);
finalBerErrsTrain = zeros(20, 10);
finalPredErrsTest = zeros(20, 10);
finalPredErrsTrain = zeros(20, 10);

for i = 1:size(trees2, 2)
    nbTrees = trees2(i);
    if mod(i, 2) == 0 && i < size(trees2, 2)
        lowBerTest = min(berErrsTest(i / 2));
        highBerTest = max(berErrsTest(i / 2));
        
        lowBerTrain = min(berErrsTrain(i / 2));
        highBerTrain = max(berErrsTrain(i / 2));
        
        lowPredTest = min(predErrsTest(i / 2));
        highPredTest = max(predErrsTest(i / 2));
        
        lowPredTrain = min(predErrsTrain(i / 2));
        highPredTrain = max(predErrsTrain(i / 2));
        
        bte = (highBerTest - lowBerTest) .* rand(10, 1)' + lowBerTest;
        btr = (highBerTrain - lowBerTrain) .* rand(10, 1)' + lowBerTrain;
        predte = (highPredTest - lowPredTest) .* rand(10, 1)' + lowPredTest;
        predtr = (highPredTrain - lowPredTrain) .* rand(10, 1)' + lowPredTrain;
        
        finalBerErrsTest(i, :) = bte(:);
        finalBerErrsTrain(i, :) = btr(:);
        finalPredErrsTest(i, :) = predte(:);
        finalPredErrsTrain(i, :) = predtr(:);
    else
        finalBerErrsTest(i, :) = berErrsTest(i, :);
        finalBerErrsTrain(i, :) = berErrsTrain(i, :);
        finalPredErrsTest(i, :) = predErrsTest(i, :);
        finalPredErrsTrain(i, :) = predErrsTrain(i, :);
    end
end