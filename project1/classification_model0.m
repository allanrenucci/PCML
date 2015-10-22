% logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
K = 5;

X = X_train;
y = y_train;

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
    
    % Set to 0 values that are -1
    yTr(yTr == -1) = 0;
    yTe(yTe == -1) = 0;
    if mean(yTr) > 0.5
        yR = 1;
    else
        yR = 0;
    end
    
    eTrSub(k) = sum(yTr * yR - log(1 + exp(yR)));
    eTeSub(k) = sum(yTe * yR - log(1 + exp(yR)));
end

eTr = mean(eTrSub);
eTe = mean(eTeSub);

%mseTr = mean(mseTrSub);
%mseTe = mean(mseTeSub);

