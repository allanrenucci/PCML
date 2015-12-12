function err = SVM(XTrain, yTrain, XTest, yTest, coeff)

% Constants
KernelFunction = 'gaussian'; % loss = 0.6
%KernelFunction = 'linear'; % loss = 0.33
Verbose = 2;
Coding = 'onevsall';

% Apply dimensionality reduction 
XTrain = XTrain * coeff;
XTest = XTest * coeff;

t = templateSVM('Standardize' ,1 , ...
    'KernelFunction', KernelFunction, ...
    'KernelScale', 'auto');

Mdl = fitcecoc(XTrain, yTrain, ...
    'Coding', Coding, ...
    'Learners', t, ...
    'ClassNames',[1 2 3 4], ...
    'Verbose', Verbose);

yPred = predict(Mdl, XTest, 'Verbose', Verbose);

predErr = sum(yPred ~= yTest) / length(yTest);
ber = BER(4, yTest, yPred);
fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErr * 100, ber);

err = [predErr, ber];

end