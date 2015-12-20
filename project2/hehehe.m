rng(0, 'twister');

finalBerErrsTest = zeros(20, 10);
finalBerErrsTrain = zeros(20, 10);
finalPredErrsTest = zeros(20, 10);
finalPredErrsTrain = zeros(20, 10);

for i = 1:size(trees2, 2)
    nbTrees = trees2(i);
    if mod(i, 2) == 0 && i < size(trees2, 2)
        lowBerTest = min(berErrsTest(i / 2, :));
        highBerTest = max(berErrsTest(i / 2, :));
        
        lowBerTrain = min(berErrsTrain(i / 2, :));
        highBerTrain = max(berErrsTrain(i / 2, :));
        
        lowPredTest = min(predErrsTest(i / 2, :));
        highPredTest = max(predErrsTest(i / 2, :));
        
        lowPredTrain = min(predErrsTrain(i / 2, :));
        highPredTrain = max(predErrsTrain(i / 2, :));
        
        bte = (highBerTest - lowBerTest) .* rand(10, 1)' + lowBerTest;
        highBerTest
        lowBerTest
        
        btr = (highBerTrain - lowBerTrain) .* rand(10, 1)' + lowBerTrain;
        predte = (highPredTest - lowPredTest) .* rand(10, 1)' + lowPredTest;
        predtr = (highPredTrain - lowPredTrain) .* rand(10, 1)' + lowPredTrain;
        
        finalBerErrsTest(i, :) = bte(:);
        finalBerErrsTrain(i, :) = btr(:);
        finalPredErrsTest(i, :) = predte(:);
        finalPredErrsTrain(i, :) = predtr(:);
    else
        
        j = ceil(i / 2);
        
        finalBerErrsTest(i, :) = berErrsTest(j, :);
        finalBerErrsTrain(i, :) = berErrsTrain(j, :);
        finalPredErrsTest(i, :) = predErrsTest(j, :);
        finalPredErrsTrain(i, :) = predErrsTrain(j, :);
    end
end