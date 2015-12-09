function err = logisticRegression(XTrain, yTrain, xTest, yTest)

[normXTrain, mu, sigma] = zscore(XTrain); % train, get mu and std
Tl = reduceDimension(normXTrain);
normXTrain = normXTrain * Tl;

normXTest = normalize(xTest, mu, sigma);  % normalize test data
normXTest = normXTest * Tl;

B = mnrfit(normXTrain, double(yTrain));
yPred = mnrval(B, normXTest);
[~, classVote] = max(yPred, [], 2);
err = sum(classVote ~= yTest) / length(yTest);

end

