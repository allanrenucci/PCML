zolTr = [0.0773 0.0818 0.0816 0.0817 0.0815 0.0815 0.0769 0.0753 0.0758 0.0758];
zolTe = [0.1067 0.0933 0.0800 0.0856 0.0733 0.0800 0.0841 0.0753 0.0843 0.0800];

plot(zolTr);
hold on;
plot(zolTe);
legend('Training error', 'Testing error', 'Location', 'northeast');
xlabel('Number of folds');
ylabel('0-1 Loss');
set(gca,'XTick', 1:10, 'XTickLabel', [2 4 6 8 10 12 14 16 18 20])