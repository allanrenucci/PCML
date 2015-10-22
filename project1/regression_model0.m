% Constant regression using mean y

clear all
load('PuntaCana_regression.mat');

% Constants
K = 5;

X = X_train;
y = y_train;

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
    yTr = y(idxTr);
    XTr = X(idxTr,:);
    
    yMean = mean(yTr);
    
    e = yTr - yMean;
    L = e' * e / (2 * length(yTr));
    
    % training and test MSE
    e = yTr - yMean;
    L = e' * e / (2 * length(yTr));
    mseTrSub(k) = sqrt(2*L);

    % testing MSE using least squares
    e = yTe - yMean;
    L = e' * e / (2 * length(yTe));
    mseTeSub(k) = sqrt(2*L);
end

mseTr = mean(mseTrSub);
mseTe = mean(mseTeSub);

