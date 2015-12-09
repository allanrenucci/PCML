load train/train.mat;

X = train.X_hog;
X = X - repmat(mean(X), size(X, 1), 1);

fprintf('Dimensionality reduction... \n');
T = reduceDimension(X);
X = X * T;
fprintf('Reduced from %d to %d features\n', size(train.X_hog, 2), size(X, 2));

csvwrite('reduced_xhog.csv', X);