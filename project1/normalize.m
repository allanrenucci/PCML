function X = normalize(X)

meanX = mean(X);
X = X - ones(size(X)) * diag(meanX);
stdX = std(X);
X = X ./ (ones(size(X)) * diag(stdX));

end

