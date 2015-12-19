clear all
load('train/train.mat');

X = train.X_cnn;

[coeff,score,latent,tsquared,explained,mu] = pca(X);

figure

hline = refline([0 100]);
hline.Color = 'r';
hline.LineStyle = '--';

hold on
x = 1:length(explained);
y = cumsum(explained);
plot(x, y);

