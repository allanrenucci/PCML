function [ encoded ] = dummyEncode(X, cols)

if isempty(cols)
    encoded = X;
else
    toEncode = X(:, cols(1));
    maximum = max(toEncode);
    minimum = min(toEncode);
    categories = maximum - minimum + 1;
    
    newColumns = zeros(size(toEncode, 1), categories);
    
    for value = minimum:maximum
        for j = 1:size(toEncode, 1)
            if toEncode(j) == value
                newColumns(value, j) = 1;
            end
        end
    end
    
    if (cols(1) > 1)
        before = X(:, 1:(cols(1) - 1));
    else
        before = [];
    end
    
    if cols(1) < size(X, 2)
        after = X(:, (cols(1) + 1):end);
    else
        after = [];
    end
    
    remainingCols = cols(2:end) + (categories - 1); 
    
    encoded = dummyEncode([before newColumns after], remainingCols);
end

end

