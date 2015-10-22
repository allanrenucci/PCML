% ridge regression

clear all
load('PuntaCana_regression.mat');

% Constants
lambda = 1;
degree = 15;
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
    
    % form tX
    tXTr = phi(XTr, degree);
	tXTe = phi(XTe, degree);
    
    beta = ridgeRegression(yTr, tXTr, lambda);
    
	% training and test MSE(INSERT CODE)
	mseTrSub(k) = sqrt(2*computeCost(yTr,tXTr,beta));

	% testing MSE using least squares
	mseTeSub(k) = sqrt(2*computeCost(yTe,tXTe,beta));
end

mseTr = mean(mseTrSub);
mseTe = mean(mseTeSub);

