function beta = ridgeRegression(y, tX, lambda)

% Number of columns - 1
M = size(tX, 2) - 1;
lIm = lambda * eye(M);
L = [zeros(1, M + 1); zeros(M, 1) lIm];

% Compute beta
beta = (tX' * tX + L) \ tX' * y;

end