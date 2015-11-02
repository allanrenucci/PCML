% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
K = 4;
X = X_train;
y = y_train;
degree = 1;

X = cleanData(X, [7 9 32], [15 26]);

alpha = 0.003;

props = linspace(0.1, 0.9, 41);

for i = 1:size(props, 2)
    fprintf('i = %d\n', i); 
    for j = 1:5
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

        % Set to 0 values that are -1
        yTr(yTr == -1) = 0;
        yTe(yTe == -1) = 0;

        % form tX
        %tXTr = [ones(size(XTr, 1), 1) XTr];
        %tXTe = [ones(size(XTe, 1), 1) XTe];
        tXTr = phi(XTr, degree);
        tXTe = phi(XTe, degree);

        beta = logisticRegression(yTr, tXTr, alpha);

        % training and test MSE(INSERT CODE)
        [rmseTrSub(i, j), loglossTrSub(i, j)] = LogisticRegressionCost(yTr, tXTr, beta);% computeRMSE(yTr, tXTr, beta);

        % testing MSE using least squares
        [rmseTeSub(i, j), loglossTeSub(i, j)] = LogisticRegressionCost(yTe, tXTe, beta); % computeRMSE(yTe, tXTe, beta);
    end
end

figure;
boxplot(loglossTrSub', props, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(loglossTeSub', props, 'colors', 'r', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
xlabel('Proportion of training data');
ylabel('RMSE');
set(gca,'XTick', 1:5:41, 'XTickLabel', [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]);