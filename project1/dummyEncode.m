function [X_dummy, X] = dummyEncode(X, cols)

X_dummy = X(:, cols);
X_dummy = dummyvar(X_dummy);
X(:, cols) = [];

end

