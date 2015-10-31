% Feature Processing with Ridge Regression

clear all
load('PuntaCana_regression.mat');

% Constants
K = 10;
D = 95; % Feature used to choose model
% Model 1: feature 95 >= 0.5
m1Degree = 1;
m1Lambda = 109.854114;
% Model 2: feature 95 < 0.5
m2Degree = 3;
m2Lambda = 2.811769;

% Dummy encode categorical variables
binVars = [39 48 49];
catVars = [11 34 40 42 50 67 72];
X_reduced = cleanData(X_train, binVars, catVars);
y = y_train;

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
    D = D - 1;
    % filter out feature one by one
    for f = 1:(size(X_reduced, 2) - 1)

        X = removeCols(X_reduced, f);

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

            % Model 1
            varsTrM1 = XTr(:, D) >= 0.5;
            XTrM1 = XTr(varsTrM1, :);
            yTrM1 = yTr(varsTrM1);
            varsTeM1 = XTe(:, D) >= 0.5;
            XTeM1 = XTe(varsTeM1, :);
            yTeM1 = yTe(varsTeM1);

            % Model 2
            varsTrM2 = XTr(:, D) < 0.5;
            XTrM2 = XTr(varsTrM2, :);
            yTrM2 = yTr(varsTrM2);
            varsTeM2 = XTe(:, D) < 0.5;
            XTeM2 = XTe(varsTeM2, :);
            yTeM2 = yTe(varsTeM2);

            % form tX
            tXTrM1 = phi(XTrM1, m1Degree);
            tXTeM1 = phi(XTeM1, m1Degree);
            tXTrM2 = phi(XTrM2, m2Degree);
            tXTeM2 = phi(XTeM2, m2Degree);

            betam1 = ridgeRegression(yTrM1, tXTrM1, m1Lambda);
            betam2 = ridgeRegression(yTrM2, tXTrM2, m2Lambda);

            % RMSE on training set
            N = length(yTr);
            eM1 = yTrM1 - tXTrM1 * betam1;
            eM2 = yTrM2 - tXTrM2 * betam2;
            rmseTrSub(k) = sqrt((eM1' * eM1 + eM2' * eM2) / N);

            % RMSE on test set
            N = length(yTe);
            eM1 = yTeM1 - tXTeM1 * betam1;
            eM2 = yTeM2 - tXTeM2 * betam2;
            rmseTeSub(k) = sqrt((eM1' * eM1 + eM2' * eM2) / N);
        end

        % Mean RMSE over K fold
        rmseTr(f) = mean(rmseTrSub);
        rmseTe(f) = mean(rmseTeSub);

    end

    % minimum error over removed features 
    [bestTe, index] = min(rmseTe);
    bestTr = min(rmseTr);

    if bestTe < best
        best = bestTe;
        X_reduced = removeCols(X_reduced, index);
        fprintf('Remove %d: rmseTr=%f, rmseTe=%f\n', index, bestTr, bestTe);
    else
        break;
    end

end
