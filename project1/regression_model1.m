% Linear regression using least square gradient descent

clear all
load('PuntaCana_regression.mat');

% Constants
alpha = 0.1;
K = 5;

X = X_train;
y = y_train;

%X = dummyEncode(X, [11 34 39 40 42 48 49 50 67 72]);

X = normalize(X);

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
    
    % form tX
    tXTr = [ones(length(yTr), 1) XTr];
    tXTe = [ones(length(yTe), 1) XTe];
    
    beta = leastSquaresGD(yTr, tXTr, alpha);
    
    % RMSE on training set
    mseTrSub(k) = computeRMSE(yTr,tXTr,beta);

    % RMSE on test set
    mseTeSub(k) = computeRMSE(yTe,tXTe,beta);
end

% Mean MSE over K fold
mseTr = mean(mseTrSub);
mseTe = mean(mseTeSub);

fprintf('mseTr=%f, mseTe=%f \n', mseTr, mseTe);

