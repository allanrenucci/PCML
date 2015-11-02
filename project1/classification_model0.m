% logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
K = 5;

X = cleanData(X_train, [7 9 32], [15 26]);
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
    
    eTrSub(k) = sum(yTr(yTr ~= yR)) / length(yTr);
    eTeSub(k) = sum(yTe(yTe ~= yR)) / length(yTe);
    
end

zolTr = mean(eTrSub);
zolTe = mean(eTeSub);

fprintf('Training error: %f\n', zolTr);
fprintf('Testing error : %f\n', zolTe);
