function err = SVM(XTrain, yTrain, XTest, yTest)

% Constants
KernelFunction = 'gaussian'; % loss = 0.6
%KernelFunction = 'linear'; % loss = 0.33
Verbose = 2;
Coding = 'onevsall';
options = statset('UseParallel', false);


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
    'Verbose', Verbose, ...
    'Options', options);fitcecoc

trPred = predict(Mdl, XTrain, 'Verbose', Verbose);
tePred = predict(Mdl, XTest, 'Verbose', Verbose);

trErr = sum(trPred ~= yTrain) / length(yTrain);
teErr = sum(tePred ~= yTest) / length(yTest);

trBer = BER(4, yTrain, trPred);
teBer = BER(4, yTest, tePred);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', teErr * 100, teBer);

err = [trErr, trBer, teErr, teBer];

end