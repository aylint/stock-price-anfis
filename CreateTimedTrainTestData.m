% function to split time series data into training and test sets

function [XTrain, XTest, YTrain, YTest] = CreateTimedTrainTestData(Inputs, Targets)

    nData = size(Inputs,1);
    nTrainData=nData-1;
    
    TrainInd=(1:nTrainData);
    XTrain=Inputs(TrainInd,:);
    YTrain=Targets(TrainInd,:);

    TestInd=(nTrainData+1:nData);
    XTest=Inputs(TestInd,:);
    YTest=Targets(TestInd,:);

end
