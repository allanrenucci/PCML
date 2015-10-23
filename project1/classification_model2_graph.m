% Penalized logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
alphaValues = 0.1:0.05:0.5;
lambdaValues = 0.1:0.05:2.0;
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

mleTr = zeros(length(alphaValues), length(lambdaValues));
mleTe = zeros(length(alphaValues), length(lambdaValues));

for i = 1:length(alphaValues)
    alpha = alphaValues(i);
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
        end

        mleTr(i, j) = mean(mleTrSub);
        mleTe(i, j) = mean(mleTeSub);
    end
end

[mesh1, mesh2] = meshgrid(alphaValues, lambdaValues);
plot3(mesh1, mesh2, mleTr, mesh1, mesh2, mleTe);
