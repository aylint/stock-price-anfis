%
% Copyright (c) 2013, Research Infinite Solutions llp (http://www.researchinfinitesolutions.com/)
% 
%
% Project Code: YPFZ101
% Project Title: Stock Market prediction using ANFIS
% Publisher:  (http://www.researchinfinitesolutions.com/)
% 
% Developer: Ruchi Mehra (Member of Research Infinite Solutions llp)
% 
% Contact Info: info@researchinfinitesolutions.com, ruchi@webtunix.com

% For training in Stock market and Data Science contact at  ::::
% ruchi@webtunix.com



function PlotResults(targets, outputs, Name)

    errors=targets-outputs;

    MSE=mean(errors.^2);
    RMSE=sqrt(MSE);
    
    error_mean=mean(errors);
    error_std=std(errors);

    subplot(2,2,[1 2]);
    plot(targets,'k');
    hold on;
    plot(outputs,'r');
    legend('Target','Output');
    title(Name);
    xlabel('Sample Index');
    grid on;

    subplot(2,2,3);
    plot(errors);
    legend('Error');
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    grid on;

    subplot(2,2,4);
    histfit(errors, 50);
    title(['Error Mean = ' num2str(error_mean) ', Error St.D. = ' num2str(error_std)]);

end