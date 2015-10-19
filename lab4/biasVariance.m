clear all;

N = 50;
degrees = 1:10;

for s = 1:200 % # of seeds
    setSeed(s);
    % generate data as shown in above matlab code
    X = linspace(0.1,2*pi,N)';
    y = sin(X(:)) + 0.3*randn(N,1);
    % randomly permute data
    idx = randperm(N);
    y = y(idx);
    X = X(idx);
    % split data
    [XTr, yTr, XTe, yTe] = split(y,X,0.9);
    
    for k = 1:length(degrees)
        degree = degrees(k);
        tXTr = [ones(length(yTr), 1) myPoly(XTr, degree)];
        tXTe = [ones(length(yTe), 1) myPoly(XTe, degree)];
        
        % get beta using least squares
        beta = leastSquares(yTr, tXTr);
        
        % compute train and test RMSE
        rmseTr(s,k) = sqrt(2*computeCost(yTr,tXTr,beta)); 
        rmseTe(s,k) = sqrt(2*computeCost(yTe,tXTe,beta));
    end
end

% compute expected train and test error
rmseTr_mean = mean(rmseTr);
rmseTe_mean = mean(rmseTe);

% plot
plot(degrees, rmseTe,'r-','color',[1 0.7 0.7]);
hold on
plot(degrees, rmseTr,'b-','color',[0.7 0.7 1]); plot(degrees, rmseTe_mean,'r-','linewidth', 3); hold on
plot(degrees, rmseTr_mean,'b-','linewidth', 3);
xlabel('degree');
ylabel('error');

boxplot(rmseTe, 'boxstyle', 'filled');