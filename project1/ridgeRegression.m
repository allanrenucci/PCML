function beta = ridgeRegression(y, tX, lambda)

degree = 7;
phiX = phi(tX, degree);
lIm = lambda * eye(degree);
L = [zeros(1, degree + 1); zeros(degree, 1) lIm];

beta = (phiX' * phiX + L) \ phiX' * y;

end