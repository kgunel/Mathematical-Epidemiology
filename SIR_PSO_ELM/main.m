%% SIR (Susceptible-Infected-Recovered) model
% S hasta veya virüse bulaþmamýþ birey sayýsý
% I virüs bulaþmýþ birey sayýsý
% R iyileþen ve/veya ölen birey sayýsý
% N = Popülasyondaki toplam birey sayýsý

%% Problem Definition
% S' = -beta*S*I/N
% I' = beta*I*S/N - gamma*I
% R' = gamma*I
% constraint : N = S + I + R
clear;clc;close;
load('data\covidData_tr_24 April 2021.mat')

%% Prepare Data 
veriSay = length([covidData_tr(:).toplam_vaka_sayisi]);
S = [covidData_tr(veriSay:-1:1).gunluk_vaka_sayisi];      % S
E = [covidData_tr(veriSay:-1:1).gunluk_test_sayisi];      % E
I = [covidData_tr(veriSay:-1:1).gunluk_hasta_sayisi];     % I
R = [covidData_tr(veriSay:-1:1).gunluk_iyilesen_sayisi];  % R
D = [covidData_tr(veriSay:-1:1).gunluk_vefat_sayisi];     % D
data = [S; E; I; R; D]';
dates = [covidData_tr(veriSay:-1:1).tarih];
t = 1:veriSay;

%% Normalize data
normalizedS = S./(S+I+R); 
normalizedI = I./(S+I+R);
normalizedR = R./(S+I+R);
normalizedN = 1;

s0 = normalizedS(1);
i0 = normalizedI(1);
r0 = normalizedR(1);

%% Prepare train and test data 
nTestData = 30;   % for weekly prediction
nDaysForRealData = floor((veriSay-nTestData)/2);
inputs_train = 1:veriSay-nTestData;
inputs_test = veriSay-nTestData+1:veriSay;

S_train = normalizedS(inputs_train);
I_train = normalizedI(inputs_train);
R_train = normalizedR(inputs_train);
S_real_train = normalizedS(inputs_train);
I_real_train = normalizedI(inputs_train);
R_real_train = normalizedR(inputs_train);

S_test = normalizedS(inputs_test);
I_test = normalizedI(inputs_test);
R_test = normalizedR(inputs_test);
S_real_test = normalizedS(inputs_test);
I_real_test = normalizedI(inputs_test);
R_real_test = normalizedR(inputs_test);

%% PSO Problem
problem.nVar = 3;                       % Number of Decision Variables, [beta, gamma # of neurons of ELM]
problem.VarMin = [0.2, -10, 1];    % Lower Bound of Variables
problem.VarMax = [2, 10, 200];     % Upper Bound of Variables
problem.normalizedN = normalizedN;
problem.input = [S_train' I_train' R_train'];

%% PSO Parameters
params.MaxIt = 100;        % Maximum Number of Iterations
params.nPop = 30;          % Population Size (Swarm Size)
params.w = 1;               % Inertia Weight
params.wdamp = 0.99;        % Inertia Weight Damping Ratio
params.c1 = 1.5;            % Personal Learning Coefficient
params.c2 = 2.0;            % Global Learning Coefficient
params.phi1 = 2.05;         % If it is used Constriction Coefficients for PSO
params.phi2 = 2.05;

%% Parameters for Extreme Learning Machine
params.VarMin = -1;       % Lower Bound of ELM weights
params.VarMax = 1;        % Upper Bound of ELM weights
params.algo = 'ELM tuned by PSO';

%% Solve SIR model
BestSol = pso(problem,params);
beta = BestSol.Position(1);
gamma = BestSol.Position(2);
params.noNeurons = BestSol.Position(3);
pBest = BestSol.netWeights;
minPerf = BestSol.Cost;

%% Results
% Trial solutions using ELM for training set
X = problem.input;
init = [s0, i0, r0];
outs = trialSolution(X, init, pBest); 
S_trial_train = outs(1,:);
I_trial_train = outs(2,:);
R_trial_train = outs(3,:);

% Trial solutions using ELM for test set
Xtest = [S_test' I_test' R_test'];
outs = trialSolution(Xtest, init, pBest); 
S_trial_test = outs(1,:);
I_trial_test = outs(2,:);
R_trial_test = outs(3,:);

%% Reports
heading = ['The numerical solution of SIR epidemic model via '  params.algo ' for training set'];
label = ['tbl:' params.algo '_train'];
displayResults(inputs_train,S_real_train,I_real_train,R_real_train,S_trial_train,I_trial_train,R_trial_train,heading,label,params);

heading = ['The numerical solution of SIR epidemic model via '  params.algo ' for test set'];
label = ['tbl:' params.algo '_test'];
displayResults(inputs_test,S_real_test,I_real_test,R_real_test,S_trial_test,I_trial_test,R_trial_test,heading,label,params);

fprintf("\nNumber of neurons in network =  %d\n",params.noNeurons);
fprintf("beta = %5.3e,\tgamma =  %5.3e\n",beta,gamma);
R0 = beta/gamma;         % Reproduction number
fprintf("Reproduction number = %5.3e\n",R0);

%% Plots
figure;
hold on;
dates = [covidData_tr(veriSay:-1:1).tarih];
plot(dates,normalizedS,'-b','LineWidth',1.5,'DatetimeTickFormat','dd/MM');
plot(dates,normalizedI,'-r','LineWidth',1.5,'DatetimeTickFormat','dd/MM');
plot(dates,normalizedR,'-k','LineWidth',1.5,'DatetimeTickFormat','dd/MM');

plot(dates(inputs_train),S_trial_train,'o','LineWidth',1.25,'MarkerSize',8,'MarkerEdgeColor',[0.1171875,0.79296875,0.87890625],'MarkerFaceColor',[1, 1, 1]);
plot(dates(inputs_train),I_trial_train,'o','LineWidth',1.25,'MarkerSize',8,'MarkerEdgeColor',[0.9453125,0.62109375,0.83984375],'MarkerFaceColor',[1, 1, 1]);
plot(dates(inputs_train),R_trial_train,'o','LineWidth',1.25,'MarkerSize',8,'MarkerEdgeColor',[0.23828125,0.8359375,0.4609375],'MarkerFaceColor',[1, 1, 1]);

plot(dates(inputs_test),S_trial_test,'x','LineWidth',1.25,'MarkerSize',12,'MarkerEdgeColor',[0.067, 0.47, 0.392],'MarkerFaceColor',[1, 1,1]);
plot(dates(inputs_test),I_trial_test,'x','LineWidth',1.25,'MarkerSize',12,'MarkerEdgeColor',[0.86, 0.506, 0],'MarkerFaceColor',[1, 1, 1]);
plot(dates(inputs_test),R_trial_test,'x','LineWidth',1.25,'MarkerSize',12,'MarkerEdgeColor',[0.25, 0.75, 0],'MarkerFaceColor',[1, 1, 1]);

plot(dates,normalizedS,'-b','LineWidth',1.5,'DatetimeTickFormat','dd/MM');
plot(dates,normalizedI,'-r','LineWidth',1.5,'DatetimeTickFormat','dd/MM');
plot(dates,normalizedR,'-k','LineWidth',1.5,'DatetimeTickFormat','dd/MM');

xlabel('\fontsize{12}\bf Dates');
ylabel('\fontsize{12}\bf The rate of Covid cases of the population (%)');
xlim([dates(1) dates(veriSay)])
xtickangle(45)
legend('Real observed data for S','          I','          R', ...
    'Training set: ELM Solution for S ','         I','         R', ...
    'Test set: Prediction of ELM for S ','         I','         R', 'Location','eastoutside');
title({'Daily Covid Cases Rates in Turkey','SIR-Epidemiology Model'});
hold off;

fig=gcf;
fig.InvertHardcopy = 'on';
saveas(fig,['figs\Figure_' params.algo '.fig']);
print(gcf,['figs\Figure_' params.algo '.jpg'],'-djpeg','-r300');
print(gcf,['figs\Figure_' params.algo '.eps'],'-depsc','-r300');
