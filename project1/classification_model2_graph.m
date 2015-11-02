% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
K = 2;
X = X_train;
y = y_train;
degrees = [1 2 3 4 5];
alpha = 0.003;
lambdaValues = logspace(-4, 2, 10);

X = cleanData(X, [7 9 32], [15 26]);

% split data in K fold (we will only create indices)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for dX = 1:length(degrees)
    fprintf('Degree = %d\n', degrees(dX));
    d = degrees(dX);
    for i = 1:length(lambdaValues)

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
            tXTr = phi(XTr, d);
            tXTe = phi(XTe, d);

            beta = penLogisticRegression(yTr, tXTr, alpha, lambdaValues(i));

            % training and test MSE(INSERT CODE)
            zolTrSub(k) = PenLogisticRegressionCost(yTr, tXTr, beta, lambdaValues(i));

            % testing MSE using least squares
            zolTeSub(k) = PenLogisticRegressionCost(yTe, tXTe, beta, lambdaValues(i));
        end

        zolTr(d, i) = mean(zolTrSub);
        zolTe(d, i) = mean(zolTeSub);

        fprintf('degree = %d, Lambda = %f\n', d, lambdaValues(i));
        fprintf('\tTraining: mle = %f, zol = %f, rmse = %f\n', 0, zolTr(d, i), 0);
        fprintf('\tTesting : mle = %f, zol = %f, rmse = %f\n', 0, zolTe(d, i), 0);
        fprintf('\tPerform : %f%%\n', (1 - zolTe(d, i)) * 100);

    end
end

plot(zolTr);
hold on;
plot(zolTe);
