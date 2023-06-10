function [perf,p] = elmSIR_train(X, Y, params)
% elmSIR_train function allows to train a single hidden layer
% feedforward network  for regression with Moore-Penrose pseudoinverse of matrix
% for SIR epidemic model
% Inputs:
%       - X:  N instances by Q attributes matrix of  training inputs;
%       - Y:  N raws and 1 attributes matrix of training targets
% Outputs:
%       - perf: performance by means of RMSE of regression
%       - p:    The weights of ELM 

%% Initialize network weights
m = params.noNeurons;
low = params.VarMin;
up = params.VarMax;
D = size(X,1);  % # of inputs
N = size(X,2);  % # of inputs
alph =  low + (up-low)*rand(1,m);
W = low + (up-low)*rand(m,N);
bias = rand(m,1)*2-1;
bias2 = rand*2-1;
s0 = X(1,1); i0 = X(1,2); r0 = X(1,3);
init = [s0, i0, r0];
p = [alph, reshape(W,1,size(W,1)*size(W,2)), bias', bias2 ];
y = trialSolution(X, init, p);
S_t = y(1,:); I_t = y(2,:); R_t = y(3,:);
%% Calculate the output of the hidden layer
z = W*X'+bias;
H = zeros(m,D);
for j = 1:D
   act = sigma(z(:,j));
   if j == 1
       dS = S_t(2) - S_t(1);
       dI = I_t(2) - I_t(1);
       dR = R_t(2) - R_t(1);
   elseif j == D
       dS = S_t(D) - S_t(D-1);
       dI = I_t(D) - I_t(D-1);
       dR = R_t(D) - R_t(D-1);
   else
       dS = (S_t(j+1) - S_t(j-1))/2;
       dI = (I_t(j+1) - I_t(j-1))/2;
       dR = (R_t(j+1) - R_t(j-1))/2;
   end
   H(:,j) = act.*(1+ (j-1)*(1 - act)).*(W(:,1)*dS + W(:,2)*dI + W(:,3)*dR) ;
end

%% Calculate the output weights beta using Moore-Penrose pseudoinverse of matrix
B = pinv(H')* Y ; 

%% Calculate the actual output 
out = (H' * B)' ;

%% Calculate the performance
perf = sqrt(mse(Y' - out));

%% Determine the network weights as outputs
alpha = B';
w = W(:,1)';
omega = W(:,2)';
mu =  W(:,3)';
p = [alpha w omega mu bias' bias2];
end