% Linear regression using least square gradient descent

clear all
load('PuntaCana_regression.mat');

% Constants
K = 10;
% Model 1: feature 95 >= 0.5
m1Degree = 1;
m1Lambda = 106.673394;
% Model 2: feature 95 < 0.5
m2Degree = 3;
m2Lambda = 2.998844;

X = X_train;
y = y_train;

% Dummy encode categorical variables
binVars = [39 48 49];
catVars = [11 34 40 42 50 67 72];
X(:, catVars) = X(:, catVars) + 1;
X = cleanData(X, binVars, catVars);

% Remove some features (2.7% reduction on RMSE: 663 -> 645)
X = removeCols(X, [1 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27 31 36 39 41 42 44 47 48 51 53 56 57 59 61 62 63 65 66 67 68 69 70 73 76 77 80 81 82 83 85 86]);

% Feature used to choose between the two models (last one in X)
D = size(X, 2);

props = linspace(0.1, 0.9, 41);

% split data in K fold (we will only create indices)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

for i = 1:size(props, 2)
    fprintf('i = %d\n', i); 
    for j = 1:30
        fprintf('\tj = %d\n', j);

        N = size(X, 1);
        idx = randperm(N);

        s = size(X, 1) * props(i);
        indexes = [floor(s) ceil(s)];

        idxTr = idx(1:indexes(1));
        idxTe = idx(indexes(2):end);
        
        XTr = X(idxTr, :);
        yTr = y(idxTr, :);

        XTe = X(idxTe, :);
        yTe = y(idxTe, :);
   
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
        rmseTrSub(i, j) = sqrt((eM1' * eM1 + eM2' * eM2) / N);

        % RMSE on test set
        N = length(yTe);
        eM1 = yTeM1 - tXTeM1 * betam1;
        eM2 = yTeM2 - tXTeM2 * betam2;
        rmseTeSub(i, j) = sqrt((eM1' * eM1 + eM2' * eM2) / N);
    end
end

figure;
boxplot(rmseTrSub', props, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(rmseTeSub', props, 'colors', 'r', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

set(gca,'XTick', 1:5:41, 'XTickLabel', [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])
