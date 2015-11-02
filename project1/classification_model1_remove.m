% logistic regression

close all
clear all
load('PuntaCana_classification.mat');

% Constants
K = 4;
X = X_train;
y = y_train;
degree = 1;

X_input = cleanData(X_train, [7 9 32], [15 26]);

alpha = 0.002;

% split data in K fold (we will only create indices)
setSeed(1);
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

lowest_zol = Inf;
removed = [];

while 1
    
    zolTe = ones(size(X, 2), 1);
    
    for col = 1:size(X_input, 2)
        fprintf('Testing col %d\n', col);
        if ismember(col, removed)
            continue
        end
        
        X = removeCols(X_input, removed); 

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
            %tXTr = [ones(size(XTr, 1), 1) XTr];
            %tXTe = [ones(size(XTe, 1), 1) XTe];
            tXTr = phi(XTr, degree);
            tXTe = phi(XTe, degree);

            beta = logisticRegression(yTr, tXTr, alpha);

            % training and test MSE(INSERT CODE)
            zolTrSub(k) = LogisticRegressionCost(yTr, tXTr, beta);

            % testing MSE using least squares
            zolTeSub(k) = LogisticRegressionCost(yTe, tXTe, beta);
        end

        zolTe(col) = mean(zolTeSub);

    end
    
    [min_zol, remove_me] = min(zolTe);
    
    if min_zol < lowest_zol
        lowest_zol = min_zol;
        fprintf('Remove feature %d to get %f%%\n', remove_me, (1 - zolTe) * 100);
        removed = [removed remove_me];
    else
        break
    end
        
end
