% logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
alpha = 0.5;
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
    
    % form tX
    tXTr = [ones(size(XTr, 1), 1) XTr];
	tXTe = [ones(size(XTe, 1), 1) XTe];
    
    beta = logisticRegression(yTr, tXTr, alpha);
    
	% training and test MSE(INSERT CODE)
	mleTrSub(k) = LogisticRegressionCost(yTr, tXTr, beta);

	% testing MSE using least squares
	mleTeSub(k) = LogisticRegressionCost(yTe, tXTe, beta);
end

mleTr = mean(mleTrSub);
mleTe = mean(mleTeSub);
