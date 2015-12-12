function err = randomForest(XTrain, yTrain, XTest, yTest, forestSize, m_try)

% Train the TreeBagger (Decision Forest).
B = TreeBagger(forestSize, XTrain, yTrain, ....
    'NumPrint', 10, ...
    'NumPredictorsToSample', m_try);

pred = str2double(B.predict(XTest));

predErr = sum(pred ~= yTest) / length(yTest);
ber = BER(4, yTest, pred);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErr * 100, ber);
err = [predErr, ber];

end