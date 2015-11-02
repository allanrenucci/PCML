figure;
boxplot(rmseTrSub', props, 'colors', 'b', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');
hold on;
boxplot(rmseTeSub', props, 'colors', 'r', 'boxstyle', 'filled', 'factorseparator', 'auto', 'factorgap', 'auto', 'jitter', 0.5, 'labelverbosity', 'majorminor', 'medianstyle', 'target', 'outliersize', 4, 'symbol', 'o');

set(gca,'XTick', 1:5:41, 'XTickLabel', [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9])