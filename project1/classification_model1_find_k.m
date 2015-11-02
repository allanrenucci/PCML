% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
Ks = 2:50;
X = X_train;
y = y_train;
degree = 1;

X = cleanData(X, [7 9 32], [15 26]);

alpha = 0.002;

for KIdx = 1:length(Ks)
    
    K = Ks(KIdx)
    clear idxCV;

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
        %tXTr = [ones(size(XTr, 1), 1) XTr];
        %tXTe = [ones(size(XTe, 1), 1) XTe];
        tXTr = phi(XTr, degree);
        tXTe = phi(XTe, degree);

        beta = logisticRegression(yTr, tXTr, alpha);

        % training and test MSE(INSERT CODE)
        zolTrSub(k) = LogisticRegressionCost(yTr, tXTr, beta);

        % testing MSE using least squares
        zolTeSub(k) = LogisticRegressionCost(yTe, tXTe, beta);
    end

    zolTr(KIdx) = mean(zolTrSub);
    zolTe(KIdx) = mean(zolTeSub);
end

plot(zolTr);
hold on;
plot(zolTe);
legend('Training error', 'Testing error');

%fprintf('Training: mle = %f, zol = %f, rmse = %f\n', 0, zolTr, 0);
%fprintf('Testing : mle = %f, zol = %f, rmse = %f\n', 0, zolTe, 0);
%fprintf('Perform : %f%%\n', 1 - zolTe);
