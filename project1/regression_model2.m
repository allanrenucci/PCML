% Ridge regression

clear all
load('PuntaCana_regression.mat');

% Constants
lambdas = logspace(-2,4,50);
degrees = [2 3 4];
K = 10;

X = X_train;
y = y_train;

% Cleaning data
binVars = [39 48 49];
catVars = [11 34 40 42 50 67 72];
X = cleanData(X, binVars, catVars);

% Feature used to choose between the two models
D = 95;

m1Vars = X(:, D) < 0.5;
X = X(m1Vars, :);
y = y(m1Vars);

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
            rmseTrSub(k) = computeRMSE(yTr,tXTr,beta);

            % RMSE on test set
            rmseTeSub(k) = computeRMSE(yTe,tXTe,beta);
        end

        % Mean RMSE over K fold
        rmseTr(j) = mean(rmseTrSub);
        rmseTe(j) = mean(rmseTeSub);
    end
    
    rmseTrDe(:, i) = rmseTr;
    rmseTeDe(:, i) = rmseTe;
    [minRMSETe, i] = min(rmseTe);
    fprintf('degree %d: minRMSETe=%f, lambda=%f\n', degree, minRMSETe, lambdas(i));
    
end

figure
subplot(2,1,1);
semilogx(lambdas, rmseTrDe);
subplot(2,1,2);
semilogx(lambdas, rmseTeDe);

