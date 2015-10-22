function beta = ridgeRegression(y, tX, lambda)

degree = size(tX, 2) - 1;
lIm = lambda * eye(degree);
L = [zeros(1, degree + 1); zeros(degree, 1) lIm];

beta = (tX' * tX + L) \ tX' * y;

end