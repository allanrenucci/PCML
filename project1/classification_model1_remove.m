% logistic regression

clear all
load('PuntaCana_classification.mat');

X_train = dummyEncode(licols(X_train, 1e-3), [13]);

last_perf = -1.0;
max_perf = 0.0;
iter = 0;

while max_perf >= last_perf
    
    iter = iter + 1;
    last_perf = max_perf;

    % Constants
    alpha = 1.0;
    K = 5;

    for remove = 1:size(X_train, 2)

        X = removeCols(X_train, remove);
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

            beta = logisticRegression(yTr, tXTr, alpha);

            % training and test MSE(INSERT CODE)
            mleTrSub(k) = LogisticRegressionCost(yTr, tXTr, beta);

            % testing MSE using least squares
            mleTeSub(k) = LogisticRegressionCost(yTe, tXTe, beta);

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

        mleTr = mean(mleTrSub);
        mleTe = mean(mleTeSub);
        perfTe = mean(perf);
        tmpPerf(remove) = perfTe;
        tmpMleTe(remove) = mleTe;
    end

    [max_perf, removeMe] = max(tmpPerf);
    [min_mleTe, removeMe2] = min(abs(tmpMleTe));
    
    if max_perf >= last_perf
        fprintf('Remove feature %d to get %f.\n', removeMe, max_perf);
        X_train = removeCols(X_train, removeMe);
        overall_perf(iter) = max_perf;
        overall_err_te(iter) = min_mleTe;
    end

end

plot(overall_perf);
hold on;
plot(overall_err_te);
