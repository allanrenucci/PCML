function [ error ] = BER(classes, y, ypred)

error = 0;

for c = 1:classes
    yc = (y == c);
    Nc = sum(y(yc));
    error = error + sum(yc & y ~= ypred) / Nc;
end

error = error / classes;

end
