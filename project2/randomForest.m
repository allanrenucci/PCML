function err = randomForest(XTrain, yTrain, XTest, yTest, forestSize, m_try)

    % Train the TreeBagger (Decision Forest).
    B = TreeBagger(forestSize, XTrain, yTrain, ....
        'Method', 'classification', ...
        'NumPrint', 10, ...
        'NumPredictorsToSample', m_try);

    pred = str2double(B.predict(XTest));
    
    predErr = sum(pred ~= yTest) / length(yTest);
    ber = BER(4, yTest, pred);

    fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErr * 100, ber);
    err = [predErr, ber];

%     numObs = size(yTrain, 1);
%     trees = cell(forestSize, 1);
%     
%     waitBar = waitbar(0, 'Training trees...');
%     
%     for i = 1:forestSize
%        
%        waitbar(i / forestSize / 2, waitBar, strcat('Training tree ', num2str(i), '...')); 
%        
%        % A subset of the rows WITH repetition.
%        selectedRows  = randi(numObs, numObs, 1);
%        
%        % Construct the sample data and results
%        sampleData    = XTrain(selectedRows, :);
%        sampleResults = yTrain(selectedRows, :);
%        
%        % Train a tree on this subset
%        trees{i} = fitctree(sampleData, sampleResults, ...
%            'NumVariablesToSample', m_try);
%     end
%     
%     classCount = size(trees{1}.ClassCount, 2);
%     
%     predictions = runOnTrees(trees, XTest, waitBar);
%     
%     close(waitBar);
%     
%     predErr = sum(predictions ~= yTest) / length(yTest);
%     ber = BER(classCount, yTest, predictions);
%     
%     err = [predErr, ber];
    
end