function err = randomForest(XTrain, yTrain, XTest, yTest, forestSize, m_try, coeff)

% Apply dimensionality reduction 
XTrain = XTrain * coeff;
XTest = XTest * coeff;

options = statset('UseParallel', true);

% Train the TreeBagger (Decision Forest).
B = TreeBagger(forestSize, XTrain, yTrain, ...
    'NVarToSample', m_try, ...
    'Options', options);
    %'NumPrint', 10, ...
    %'NumPredictorsToSample', m_try);

predTrain = str2double(B.predict(XTrain));
predTest = str2double(B.predict(XTest));

predErrTrain = sum(predTrain ~= yTrain) / length(yTrain);
berTrain = BER(4, yTrain, predTrain);

predErrTest = sum(predTest ~= yTest) / length(yTest);
berTest = BER(4, yTest, predTest);

fprintf('Training error: %.2f%%, ber=%f\n', predErrTrain * 100, berTrain);
fprintf('Testing error : %.2f%%, ber=%f\n\n', predErrTest * 100, berTest);
err = [predErrTrain, berTrain, predErrTest, berTest];

end