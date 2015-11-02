subplot(2, 1, 1)
semilogx(lambdaValues, zolTr)
legend('d = 1', 'd = 2', 'd = 3', 'd = 4', 'd = 5', 'Location', 'northwest');
xlabel('Lambda');
ylabel('0-1 Loss');
title('Training error');

subplot(2, 1, 2)
semilogx(lambdaValues, zolTe)
legend('d = 1', 'd = 2', 'd = 3', 'd = 4', 'd = 5', 'Location', 'northwest');
xlabel('Lambda');
ylabel('0-1 Loss');
title('Testing error');
