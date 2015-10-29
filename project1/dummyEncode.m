function X = dummyEncode(X, cols)

counter = 0;
for i = 1:size(cols, 2)
    c = cols(i) + counter;

    encodeMe = X(:, c);
    encoded = dummyvar(encodeMe);
    
    before = X(:, 1:(c - 1));
    after = X(:, (c + 1):end);

    X = [before encoded after];
    counter = counter + size(encoded, 2) - 1;
end

end

