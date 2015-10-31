% Penalized logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
alpha = 0.001;
lambdaValues = 0.0356; %logspace(-2,2,30);

degree = 3;
K = 5;

X(:, [7 9 32]) = []
[X_binary, X_train] = [X_train(:, [7 9 32]), X_train(:, 1:32 - [7 9 32]);
[X_dummy, X] = dummyEncode(X_train, [7 9 15 26 32]);
X = X_train; %licols(dummyEncode2(X_train, [7 9 15 26 32]), 1e-3);
y = y_train;

X = [X_dummy normalize(X)];

% split data in K fold (we will only create indices)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for j = 1:length(lambdaValues)
    lambda = lambdaValues(j);

    mleTrSub = zeros(k, 1);
    mleTeSub = zeros(k, 1);

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
        tXTr = phi(XTr, degree); %[ones(size(XTr, 1), 1) XTr];
        tXTe = phi(XTe, degree); %[ones(size(XTe, 1), 1) XTe];

        beta = penLogisticRegression(yTr, tXTr, alpha, lambda);

        % training and test MSE(INSERT CODE)
        [mleTrSub(k), zolTrSub(k), rmseTrSub(k)] = PenLogisticRegressionCost(yTr, tXTr, beta, lambda);

        % testing MSE using least squares
        [mleTeSub(k), zolTeSub(k), rmseTeSub(k)] = PenLogisticRegressionCost(yTe, tXTe, beta, lambda);

    end

    mleTrX(j) = mean(mleTrSub);
    zolTrX(j) = mean(zolTrSub);
    rmseTrX(j) = mean(rmseTrSub);
    mleTeX(j) = mean(mleTeSub);
    zolTeX(j) = mean(zolTeSub);
    rmseTeX(j) = mean(rmseTeSub);

end

mleTr = max(mleTrX);
zolTr = max(zolTrX);
rmseTr = max(rmseTrX);

mleTe = max(mleTeX);
zolTe = max(zolTeX);
rmseTe = max(rmseTeX);

fprintf('Training: mle = %f, zol = %f, rmse = %f\n', mleTr, zolTr, rmseTr);
fprintf('Testing : mle = %f, zol = %f, rmse = %f\n', mleTe, zolTe, rmseTe);
fprintf('Perform : %f%%\n', 1 - zolTe);

%[mesh1, mesh2] = meshgrid(alphaValues, lambdaValues);
%plot(mesh1, mesh2, mleTr, mesh1, mesh2, mleTe);
%plot(lambdaValues, performance)
