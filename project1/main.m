clear all

%load('PuntaCana_classification.mat');
load('PuntaCana_regression.mat');

X = X_train;
y = y_train;

%y(y == -1) = 0;
X = normalize(X);

figure;
title('col i and y_train');

D = size(X, 2);
for i= 1:D
     subplot(6,11,i);
     plot(X_train(:,i),y_train,'b.');
     hold on;
     %legend(['col ' num2str(i) ''], 'y');
end
