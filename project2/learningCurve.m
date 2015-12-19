function [errors, predErrsTrain, berErrsTrain, predErrsTest, berErrsTest] = learningCurve(fun, X, y, props)

N = size(X, 1);

for i = 1:size(props, 2)
    for j = 1:5
        fprintf('Training proportion = %.2f%% [%d/5]\n', props(i) * 100, j);
        idx = randperm(N);
        
        s = size(X, 1) * props(i);
        indexes = [floor(s) ceil(s)];

        idxTr = idx(1:indexes(1));
        idxTe = idx(indexes(2):end);

        XTr = X(idxTr, :);
        yTr = y(idxTr, :);

        XTe = X(idxTe, :);
        yTe = y(idxTe, :);
        
        errors(i, j, :) = fun(XTr, yTr, XTe, yTe);
        predErrsTrain(i, j) = errors(i, j, 1);
        berErrsTrain(i, j) = errors(i, j, 2);
        predErrsTest(i, j) = errors(i, j, 3);
        berErrsTest(i, j) = errors(i, j, 4);
    end
end
