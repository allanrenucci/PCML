function beta = ridgeRegression(y, tX, lambda)

M = size(tX, 2) - 1;
lIm = lambda * eye(M);
L = [zeros(1, M + 1); zeros(M, 1) lIm];

beta = (tX' * tX + L) \ tX' * y;

end