% Feature Processing with Ridge Regression

clear all
load('PuntaCana_regression.mat');

% Constants
alpha = 0.1;
lambda = 1;
K = 5;
degree = 6;

% split data in K fold (we will only create indices)
setSeed(1);
N = size(y_train,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

best = Inf;

while 1
    
    % filter out feature one by one
    for f = 1:size(X_train, 2)

        X = removeCols(X_train, f);
        y = y_train;

        X = normalize(X);

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

            % MSE on training set
            mseTrSub(k) = computeRMSE(yTr,tXTr,beta);

            % MSE on test set
            mseTeSub(k) = computeRMSE(yTe,tXTe,beta);
        end

        % Mean MSE over K fold
        mseTr(f) = mean(mseTrSub);
        mseTe(f) = mean(mseTeSub);

        end

    % minimum error over removed features 
    [bestTe, index] = min(mseTe);
    bestTr = min(mseTr);

    if bestTe < best
        best = bestTe;
        X_train = removeCols(X_train, index);
        fprintf('Remove %d: mseTr=%f, mseTe=%f\n', index, bestTr, bestTe);
    else
        break;
    end

end
