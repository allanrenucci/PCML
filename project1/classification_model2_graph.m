% Penalized logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
alpha = 0.1;
lambdaValues = logspace(-2,2,100);
K = 5;
X_dummy = dummyEncode2(X_train, [7 9 15 26 32]);
X = dummyEncode2(X_train, [7 9 15 26 32]);
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

mleTr = zeros(1, length(lambdaValues));
mleTe = zeros(1, length(lambdaValues));

for j = 1:length(lambdaValues)
    lambda = lambdaValues(j);

    mleTrSub = zeros(k, 1);
    mleTeSub = zeros(k, 1);

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

        beta = penLogisticRegression(yTr, tXTr, alpha, lambda);

        % training and test MSE(INSERT CODE)
        mleTrSub(k) = PenLogisticRegressionCost(yTr, tXTr, beta, lambda);

        % testing MSE using least squares
        mleTeSub(k) = PenLogisticRegressionCost(yTe, tXTe, beta, lambda);
        
        %%%
        %%%
        %%%
        correct = 0;
        for n = 1:length(yTe)
            tmp = tXTe(n, :) * beta;
            if tmp < 0.5
                tmp = 0;
            else
                tmp = 1;
            end
            %fprintf('Predicted = %d, actual = %d\n', tmp, yTe(n));
            if tmp == yTe(n)
                correct = correct + 1;
            end
        end
        %%%
        %%%
        %%%
        
        perf(k) = correct / length(yTe);
    end

    mleTr(j) = mean(mleTrSub);
    mleTe(j) = mean(mleTeSub);
    
    m = mean(perf)
    performance(j) = m;
end

%[mesh1, mesh2] = meshgrid(alphaValues, lambdaValues);
%plot(mesh1, mesh2, mleTr, mesh1, mesh2, mleTe);
plot(lambdaValues, performance)
