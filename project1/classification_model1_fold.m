% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
X = X_train;
y = y_train;
degree = 1;

X = cleanData(X, [7 9 32], [15 26]);

alpha = 0.003;

for K = 2:20
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
        zolTrSub(K, k) = LogisticRegressionCost(yTr, tXTr, beta);

        % testing MSE using least squares
        zolTeSub(K, k) = LogisticRegressionCost(yTe, tXTe, beta);
    end

    zolTr(K) = mean(zolTrSub(K));
    zolTe(K) = mean(zolTeSub(K));
end

plot(zolTr);
hold on;
plot(zolTe);
legend('Training error', 'Testing error', 'Location', 'northeast');
xlabel('Number of folds');
ylabel('0-1 Loss');
set(gca,'XTick', 2:20, 'XTickLabel', 2:20)
