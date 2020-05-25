clear;clc;

%% init

yhat = [];
TrainingErrors = [];

%% load text data

T = readtable('GSPC-snp500.csv');

close_data = T{end-1050:end,5};
x = close_data';

%% the main for loop: iterate through dates
% 2 months test

TEST_DAY = 7; %42

for day = TEST_DAY:-1:1
    x = close_data(end-day-999:end-day+1)';

    %% prepare data

    % model 1, weekly data
    Delays = [5 10 15 20 25];
    [InputsW, TargetsW] = CreateTimeSeriesData(x, Delays);

    %% model2: daily data
    Delays = [1 2 3 4 5];
    [InputsD, TargetsD] = CreateTimeSeriesData(x, Delays);


    %% split train & test

    [XTrainW, XTestW, YTrainW, YTestW] = CreateTimedTrainTestData(InputsW', TargetsW');
    [XTrainD, XTestD, YTrainD, YTestD] = CreateTimedTrainTestData(InputsD', TargetsD');


    %% FIS Generation

    % 1: 'Grid Partitioning (genfis1)';
    opt = genfisOptions('GridPartition');
    opt.NumMembershipFunctions=5;
    opt.InputMembershipFunctionType="gaussmf";
    InputMF='gaussmf';
    OutputMF='linear';

    %fis1=genfis1([TrainInputs TrainTargets],NumMembershipFunctions,InputMF,OutputMF);
    fis1D = genfis(XTrainD, YTrainD, opt);
    fis1W = genfis(XTrainW, YTrainW, opt);
    %showrule(fis1)

    %% tune fis 1

    [in,out,rule] = getTunableSettings(fis1D);
    opt1 = tunefisOptions("Method","anfis");
    fisout1D = tunefis(fis1D,[in;out],XTrainD,YTrainD,opt1);


    [in,out,rule] = getTunableSettings(fis1W);
    opt1 = tunefisOptions("Method","anfis");
    fisout1W = tunefis(fis1W,[in;out],XTrainW,YTrainW,opt1);


    %%
    % 2: 'Subtractive Clustering (genfis2)';
    Radius=0.55;
    %fis2=genfis2(TrainInputs,TrainTargets,Radius);

    opt = genfisOptions('SubtractiveClustering',...
                        'ClusterInfluenceRange',Radius);
    fis2D = genfis(XTrainD, YTrainD, opt);
    fis2W = genfis(XTrainW, YTrainW, opt);
    %showrule(fis2)

    %% tune fis 2

    [in,out,rule2] = getTunableSettings(fis2D);
    opt2 = tunefisOptions("Method","anfis");
    fisout2D = tunefis(fis2D,[in;out],XTrainD,YTrainD,opt2);

    [in,out,rule2] = getTunableSettings(fis2W);
    opt2 = tunefisOptions("Method","anfis");
    fisout2W = tunefis(fis2W,[in;out],XTrainW,YTrainW,opt2);

    %%
    % 3: 'FCM (genfis3)';
    nCluster=5;
    Exponent=2;
    MaxIt=100;
    MinImprovment=1e-5;
    DisplayInfo=1;
    FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];

    opt = genfisOptions('FCMClustering','FISType','sugeno');
    opt.NumClusters = 5;

    % fis3=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);
    fis3D = genfis(XTrainD, YTrainD, opt);
    fis3W = genfis(XTrainW, YTrainW, opt);
    % showrule(fis3)

    %% tune fis 3
    [in,out,rule3] = getTunableSettings(fis3D);
    opt3 = tunefisOptions("Method","anfis");
    fisout3D = tunefis(fis3D,[in;out],XTrainD,YTrainD,opt3);

    [in,out,rule3] = getTunableSettings(fis3W);
    opt3 = tunefisOptions("Method","anfis");
    fisout3W = tunefis(fis3W,[in;out],XTrainW,YTrainW,opt3);


    %% Train ANFIS
    MaxEpoch=50;
    ErrorGoal=0;
    InitialStepSize=0.01;
    StepSizeDecreaseRate=0.9;
    StepSizeIncreaseRate=1.1;
    TrainOptions=[MaxEpoch ...
                  ErrorGoal ...
                  InitialStepSize ...
                  StepSizeDecreaseRate ...
                  StepSizeIncreaseRate];
    DisplayInfo=true;
    DisplayError=true;
    DisplayStepSize=true;
    DisplayFinalResult=true;
    DisplayOptions=[DisplayInfo ...
                    DisplayError ...
                    DisplayStepSize ...
                    DisplayFinalResult];
    OptimizationMethod=1;
    % 0: Backpropagation
    % 1: Hybrid

    %% optmethod = 1
    %no tuning            
    anfis1d=anfis([XTrainD YTrainD],fis1D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2d=anfis([XTrainD YTrainD],fis2D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis3d=anfis([XTrainD YTrainD],fis3D,TrainOptions,DisplayOptions,[],OptimizationMethod);

    anfis1w=anfis([XTrainW YTrainW],fis1W,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2w=anfis([XTrainW YTrainW],fis2W,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis3w=anfis([XTrainW YTrainW],fis3W,TrainOptions,DisplayOptions,[],OptimizationMethod);

    %tuned            
    anfis1dt=anfis([XTrainD YTrainD],fisout1D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2dt=anfis([XTrainD YTrainD],fisout2D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis3dt=anfis([XTrainD YTrainD],fisout3D,TrainOptions,DisplayOptions,[],OptimizationMethod);

    anfis1wt=anfis([XTrainW YTrainW],fisout1W,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2wt=anfis([XTrainW YTrainW],fisout2W,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis3wt=anfis([XTrainW YTrainW],fisout3W,TrainOptions,DisplayOptions,[],OptimizationMethod);


    %% Apply ANFIS to Data

    % daily models
    OutputsD = zeros(6,size(InputsD,2));

    OutputsD(1,:)=evalfis(InputsD,anfis1d);
    OutputsD(2,:)=evalfis(InputsD,anfis2d);
    OutputsD(3,:)=evalfis(InputsD,anfis3d);

    OutputsD(4,:)=evalfis(InputsD,anfis1dt);
    OutputsD(5,:)=evalfis(InputsD,anfis2dt);
    OutputsD(6,:)=evalfis(InputsD,anfis3dt);

    TrainOutputsD=OutputsD(:, 1:end-1);
    TestOutputsD=OutputsD(:, end);

    % weekly models
    OutputsW = zeros(6,size(InputsW,2));

    OutputsW(1,:)=evalfis(InputsW,anfis1w);
    OutputsW(2,:)=evalfis(InputsW,anfis2w);
    OutputsW(3,:)=evalfis(InputsW,anfis3w);

    OutputsW(4,:)=evalfis(InputsW,anfis1wt);
    OutputsW(5,:)=evalfis(InputsW,anfis2wt);
    OutputsW(6,:)=evalfis(InputsW,anfis3wt);

    TrainOutputsW=OutputsW(:, 1:end-1);
    TestOutputsW=OutputsW(:, end);

    %% generate ensamble output
    nmin = min(size(TrainOutputsD,2),size(TrainOutputsW,2));
    TrainOutputsD = TrainOutputsD(4:6,end-nmin+1:end);
    TrainOutputsW = TrainOutputsW(4:6,end-nmin+1:end);
    TrainOutputs = [TrainOutputsD; TrainOutputsW];

    TestOutputs = [TestOutputsD; TestOutputsW];

    yhat = [yhat; TestOutputs' mean(TestOutputs)];
    
    TrainOutputs = mean(TrainOutputs)';
    TestOutputs = mean(TestOutputs)';

    %% Error Calculation
    nmin = min(size(YTrainD,1),size(YTrainW,1));
    TrainTargets = YTrainD(end-nmin+1:end,1);
    TestTargets = YTestD;

    TrainErrors=TrainTargets-TrainOutputs;
    TrainMSE=mean(TrainErrors.^2);
    TrainRMSE=sqrt(TrainMSE);
    TrainErrorMean=mean(TrainErrors);
    TrainErrorSTD=std(TrainErrors);
    TestErrors=TestTargets-TestOutputs;
    TestMSE=mean(TestErrors.^2);
    TestRMSE=sqrt(TestMSE);
    TestErrorMean=mean(TestErrors);
    TestErrorSTD=std(TestErrors);

    TrainingErrors = [TrainingErrors; TrainMSE TrainRMSE TrainErrorMean TrainErrorSTD];
end


%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

figure;
dailyens = yhat(:,3);
PlotResults(x(end-41:end)',dailyens,'Test Data');

figure;
PlotResults(TargetsD,Outputs,'All Data');

if ~isempty(which('plotregression'))
    figure;
    plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
                   TestTargets, TestOutputs, 'Test Data', ...
                   TargetsD, Outputs, 'All Data');
    set(gcf,'Toolbar','figure');
end


%% plot more results

x = 1:42;
% training errors
plot(x', TrainingErrors(:,1));
plot(x', TrainingErrors(:,2));


% test errors
figure
plot(x', dailyend);

title('Test Errors')
xlabel('days')
ylabel('predicted')
legend('1d','2d','3d','1dt','2dt','3dt', '1w', '2w', '3w', '1wt', '2wt', '3wt', 'ens');

avg = mean(yhat);

%% averages
yhat(:,14)=dailyens;
TestTargets = x(end-41:end)';
testerr=[];
for i=1:14
    TestErrors=TestTargets-yhat(:,i);
    TestMSE=mean(TestErrors.^2);
    TestRMSE=sqrt(TestMSE);
    TestErrorMean=mean(TestErrors);
    TestErrorSTD=std(TestErrors);
    Rsq = 1 - sum( ( TestTargets-yhat(:,i) ).^2) / sum((TestTargets - mean(TestTargets)).^2);
    
    disp(i);
    testerr = [testerr; Rsq TestMSE TestRMSE TestErrorMean TestErrorSTD ];
end