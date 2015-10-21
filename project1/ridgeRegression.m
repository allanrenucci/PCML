function beta = ridgeRegression(y, tX, lambda)

degree = 4;
phiX = tX; %phi(tX, degree);
lIm = lambda * eye(degree);
L = [zeros(1, degree + 1); zeros(degree, 1) lIm];

beta = (phiX' * phiX + L) \ phiX' * y;

end