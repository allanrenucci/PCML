% Ridge regression

clear all
load('PuntaCana_regression.mat');

% Constants
lambdas = logspace(-4,3,50);
degrees = [1 4 5 6];
K = 3;

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

for i = 1:length(degrees)
    degree = degrees(i);
    
    for j = 1:length(lambdas)
        lambda = lambdas(j);

        % K fold
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

            % Compute Beta
            beta = ridgeRegression(yTr, tXTr, lambda);

            % RMSE on training set
            mseTrSub(k) = computeRMSE(yTr,tXTr,beta);

            % RMSE on test set
            mseTeSub(k) = computeRMSE(yTe,tXTe,beta);
        end

        % Mean RMSE over K fold
        mseTr(j) = mean(mseTrSub);
        mseTe(j) = mean(mseTeSub);
    end
    
    mseTrDe(:, i) = mseTr;
    mseTeDe(:, i) = mseTe;
    
end

figure
subplot(2,1,1);
semilogx(lambdas, mseTrDe);
subplot(2,1,2);
semilogx(lambdas, mseTeDe);

