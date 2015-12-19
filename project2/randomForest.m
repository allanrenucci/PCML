function err = randomForest(XTrain, yTrain, XTest, yTest, forestSize, m_try, coeff)

options = statset('UseParallel', true);

% Apply dimensionality reduction 
XTrain = XTrain * coeff;
XTest = XTest * coeff;

% Train the TreeBagger (Decision Forest).
B = TreeBagger(forestSize, XTrain, yTrain, ...
    'NumPrint', 10, ...
    'NumPredictorsToSample', m_try, ...
    'Options', options);

pred = str2double(B.predict(XTest));

predErr = sum(pred ~= yTest) / length(yTest);
ber = BER(4, yTest, pred);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErr * 100, ber);
err = [predErr, ber];

end