% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
K = 4;

indep = X_train;
%[indep, ~] = licols(X_train, 1e-3);
%[ indep, ~ ] = licols(dummyEncode2(X_train, [7 9 15 26 32]), 1e-3);
%[indep, y_train] = removeOutliers(indep, y_train);
%indep = dummyEncode2(X_train, [7 9 15 26 32]);

X = indep; %dummyEncode(X_train, [7 9 15 26 32]);
y = y_train;

X = normalize(X);

alphaValues = 0.001;

for n = 1:length(alphaValues)
    
    alpha = alphaValues(n);

    % split data in K fold (we will only create indices)
    setSeed(1);
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end

    for k = 1:K
        % get k'th subgroup in test, others in train
        idxTe = idxCV(k,:);
        idxTr = idxCV([1:k-1 k+1:end],:);
        idxTr = idxTr(:);
        yTe = y(idxTe);
        XTe = X(idxTe,:);
        yTr = y(idxTr);
        XTr = X(idxTr,:);

        % Set to 0 values that are -1
        yTr(yTr == -1) = 0;
        yTe(yTe == -1) = 0;

        % form tX
        tXTr = [ones(size(XTr, 1), 1) XTr];
        tXTe = [ones(size(XTe, 1), 1) XTe];

        beta = logisticRegression(yTr, tXTr, alpha);

        % training and test MSE(INSERT CODE)
        [mleTrSub(k), zolTrSub(k), rmseTrSub(k)] = LogisticRegressionCost(yTr, tXTr, beta);

        % testing MSE using least squares
        [mleTeSub(k), zolTeSub(k), rmseTeSub(k)] = LogisticRegressionCost(yTe, tXTe, beta);
    end

    mleTr(n) = mean(mleTrSub);
    zolTr(n) = mean(zolTrSub);
    rmseTr(n) = mean(rmseTrSub);
    mleTe(n) = mean(mleTeSub);
    zolTe(n) = mean(zolTeSub);
    rmseTe(n) = mean(rmseTeSub);
    
end

fprintf('Training: mle = %f, zol = %f, rmse = %f\n', mleTr, zolTr, rmseTr);
fprintf('Testing : mle = %f, zol = %f, rmse = %f\n', mleTe, zolTe, rmseTe);
fprintf('Perform : %f%%\n', 1 - zolTe);
