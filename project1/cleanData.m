function X = cleanData( X, binVars, catVars )

Xbin = X(:, binVars);
Xcat = X(:, catVars);
Xcat = dummyvar(Xcat);
X(:, [binVars catVars]) = [];
X = [Xbin Xcat normalize(X)];

end

