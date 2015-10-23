function [ clean ] = removeCols(in, cols)
clean = in;
clean(:, cols) = [];
end

