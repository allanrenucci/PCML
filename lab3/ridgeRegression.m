function beta = ridgeRegression(y, tX, lambda)

    ss = size(tX);
    M = ss(2) - 1;
    im = [zeros(1, M + 1); zeros(M, 1) (lambda * eye(M))];
    beta = inv(tX' * tX + im) * tX' * y;
    %beta = (tX' * tX + lambda * eye(M)) \ (tX' * y);
end

