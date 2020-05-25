%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPFZ102
% Project Title: Time-Series Prediction using ANFIS
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [X, Y] = CreateTimeSeriesData(x, Delays)

    T = size(x,2);
    
    MaxDelay = max(Delays);
    
    Range = MaxDelay+1:T;
    
    X= [];
    for d = Delays
        X=[X; x(:,Range-d)];
    end
    
    Y = x(:,Range);

end
