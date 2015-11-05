clear all;
close all;

% read data
X = normalize(im2double(imread('lena.png')));
% data is DxN
X = X';

% center data
% WRITE CODE TO CENTER THE DATA
mX = mean(X(1, :));
mY = mean(X(2, :));
X(1, :) = X(1, :) - mX;
X(2, :) = X(2, :) - mY;

% plot data
plot(X(1,:), X(2,:), 'ko', 'markersize', 5, 'markerfacecolor', 'k');
grid on;
%axis([-2.5 2.5 -2.5 2.5]);
%set(gca, 'xtick', [-2:2],'ytick', [-2:2])
fprintf('press any key to continue...');
pause;

% initialize
%MuOld = [1 -1; -1 1];
%MuOld = [1 -1 0; -1 1 0];
MuOld = rand(size(X, 1), size(X, 2)); %[-1 1; -1 -1];
K = size(MuOld,2);
colors = {'r','b','g','c','k','m','y'};
maxIters = 10;

% iterate
for i = 1:maxIters
  % update R and Mu
  % COMPLETE THIS FUNCTION
  [Ln, r, Mu] = kmeansUpdate(X,MuOld);

  % average distance over all n
  L(i) = mean(Ln);
  fprintf('%d %f\n', i, L(i));

  % convergence
  % WRITE CODE FOR CONVERGENCE
  if sum(abs(MuOld - Mu)) < 1e-5
      break
  end

  % visualize clusters
  hold off;
  %plotClusters(X, r, MuOld, colors)
  pause(.5);
  hold off;
  %plotClusters(X, r, Mu, colors)
  pause(.5);

  % new mean is the old mean now
  MuOld = Mu;
  Lold = L(i);
end

imshow(X');

