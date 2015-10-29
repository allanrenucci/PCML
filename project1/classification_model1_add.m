% logistic regression

clear all
load('PuntaCana_classification.mat');

X_train = dummyEncode2(licols(X_train, 1e-3), [13]);

last_perf = -1.0;
max_perf = 0.0;
last_err_te = Inf;
best_err_te = Inf;
last_err_tr = Inf;
min_err_tr = Inf;

iter = 0;

selectedColumns = [];

while abs(last_err_te) <= abs(best_err_te)
    
    best_err_te = last_err_te;
    last_perf = max_perf;

    % Constants
    alpha = 1.0;
    K = 5;
    
    tmpPerf = zeros(1, size(X_train, 2)) - 1;
    tmpMleTe = zeros(1, size(X_train, 2)) - 1;
    tmpMleTr = zeros(1, size(X_train, 2)) - 1;
    
    for add = 1:size(X_train, 2)
        
        if ismember(add, selectedColumns)
            continue;
        end
        
        tmpSelected = [selectedColumns add];
        X = X_train(:, tmpSelected);
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
        tmpMleTr(add) = mleTr;
        tmpMleTe(add) = mleTe;
        tmpPerf(add) = perfTe;
    end

    [max_perf, addMe] = max(tmpPerf);
    [last_err_te, addMe2] = min(tmpMleTe);
    [min_err_tr, addMe3] = min(tmpMleTr);
    %selectedColumns
    %addMe
    
    if abs(last_err_te) <= abs(best_err_te) && ~(ismember(addMe2, selectedColumns))
        iter = iter + 1;
        fprintf('Add feature %d to get %f%% ((%d, %f TE), (%d, %f TR)).\n', addMe, max_perf * 100, addMe2, last_err_te, addMe3, min_err_tr);
        selectedColumns = [selectedColumns addMe2];
        overall_err_te(iter) = last_err_te;
        overall_perf(iter) = max_perf;
    else
        fprintf('Cannot do better than current best error: %f > %f\n', last_err_te, best_err_te);
    end

end

plot(overall_perf);
hold on;
plot(overall_err_te);
