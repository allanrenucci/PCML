clear all

load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

figure;
D = size(X, 2);
for i= 1:D
     subplot(6,6,i);
     hist(X_test(:, i));
     hold on;
end
