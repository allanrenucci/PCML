function err = SVM(XTrain, yTrain, XTest, yTest)

% Constants
KernelFunction = 'gaussian'; % loss = 0.6
%KernelFunction = 'linear'; % loss = 0.33
Verbose = 2;
Coding = 'onevsone';

% Apply dimensionality reduction 
[XTrain, mu, sigma] = zscore(XTrain);
XTest = normalize(XTest, mu, sigma);

t = templateSVM('Standardize' ,1 , ...
    'KernelFunction', KernelFunction, ...
    'KernelScale', 'auto');

Mdl = fitcecoc(XTrain, yTrain, ...
    'Coding', Coding, ...
    'Learners', t, ...
    'ClassNames',[1 2 3 4], ...
    'Verbose', Verbose);

train.pred = predict(Mdl, XTrain, 'Verbose', Verbose);
test.pred = predict(Mdl, XTest, 'Verbose', Verbose);

train.err = sum(train.pred ~= yTrain) / length(yTrain);
test.err = sum(test.pred ~= yTest) / length(yTest);

train.ber = BER(4, yTrain, train.pred);
test.ber = BER(4, yTest, test.pred);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', test.err * 100, test.ber);

err = [train.err, train.ber, test.err, test.ber];

end