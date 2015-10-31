function X = cleanData( X, binVars, catVars )

Xbin = X(:, binVars);
X(:, catVars) = X(:, catVars) + 1;
Xcat = X(:, catVars) + 1;
Xcat = dummyvar(Xcat);
X(:, [binVars catVars]) = [];
X = [Xbin Xcat normalize(X)];

end

