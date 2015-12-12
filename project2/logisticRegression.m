function err = logisticRegression(XTrain, yTrain, XTest, yTest, coeff)

% Apply dimensionality reduction 
XTrain = XTrain * coeff;
XTest = XTest * coeff;

fprintf('Training logistic regression \n');
B = mnrfit(XTrain, double(yTrain));

fprintf('Computing predictions \n');
yPred = mnrval(B, XTest);
[~, classVote] = max(yPred, [], 2);

predErr = sum(classVote ~= yTest) / length(yTest);
ber = BER(4, yTest, yPred);

err = [predErr, ber];

end

