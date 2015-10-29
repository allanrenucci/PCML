% Penalized logistic regression

clear all
load('PuntaCana_classification.mat');

% Constants
alpha = 0.5;
lambda = 1.0;
K = 5;

last_perf = -1.0;
max_perf = 0.0;
iter = 0;

while max_perf >= last_perf
    
    iter = iter + 1;
    last_perf = max_perf;

    for remove = 1:size(X_train, 2)

        X = dummyEncode2(X_train, [7 9 15 26 32]);
        X = removeCols(X, remove);
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

        tmpPerf(remove) = mean(perf);
        tmpErrTe(remove) = mean(mleTeSub);
        
    end
    
    [max_perf, removeMe] = max(tmpPerf);
    [min_errTe, remomoveMe2] = min(tmpErrTe);
    
    if max_perf >= last_perf
        fprintf('Remove feature %d to get %f.\n', removeMe, max_perf);
        X_train = removeCols(X_train, removeMe);
        overall_perf(iter) = max_perf;
        overall_errTe(iter) = min_errTe;
    end
    
end

% Features that are removed: 11, 9, 12, 5, 15, 16, 23

plot(overall_errTe);
hold on;
plot(overall_perf);
