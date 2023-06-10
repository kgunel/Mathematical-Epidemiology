function BestSol = pso(problem,params)
%% Problem Definition
CostFunction = @(X,Y,params) elmSIR_train(X, Y, params); % Cost Function

nVar = problem.nVar;             % Number of Decision Variables
VarSize = [1 nVar];              % Size of Decision Variables Matrix
VarMin = problem.VarMin;        
VarMax = problem.VarMax;         
X = problem.input;               % inputs
%% PSO Parameters
MaxIt = params.MaxIt;      % Maximum Number of Iterations
nPop = params.nPop;        % Population Size (Swarm Size)
w = params.w;              % Inertia Weight
wdamp = params.wdamp;      % Inertia Weight Damping Ratio
c1 = params.c1;            % Personal Learning Coefficient
c2 = params.c2;            % Global Learning Coefficient

% If you would like to use Constriction Coefficients for PSO, 
% uncomment the following block and comment the above set of parameters.
% % Constriction Coefficients
% phi1 = 2.05;
% phi2 = 2.05;
% phi = phi1+phi2;
% chi = 2/(phi-2+sqrt(phi^2-4*phi));
% w = chi;          % Inertia Weight
% wdamp = 1;        % Inertia Weight Damping Ratio
% c1 = chi*phi1;    % Personal Learning Coefficient
% c2 = chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax = floor(0.1*(VarMax-VarMin));
VelMin = -VelMax;

%% Initialization

empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.Velocity = [];
empty_particle.netWeights = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

particle = repmat(empty_particle, nPop, 1);

GlobalBest.Cost = inf;

for i = 1:nPop
    
    % Initialize Position
    for j=1:nVar
        particle(i).Position(j) = unifrnd(VarMin(j), VarMax(j), 1);
    end
    particle(i).Position(nVar) = floor(particle(i).Position(nVar)); 
    % Initialize Velocity
    particle(i).Velocity = zeros(VarSize);
    
    % Evaluation
    beta = particle(i).Position(1);
    gamma = particle(i).Position(2);
    dS = -beta*problem.input(:,1).*problem.input(:,2)./problem.normalizedN;
    dI = beta*problem.input(:,1).*problem.input(:,2)./problem.normalizedN - gamma.*problem.input(:,2);
    dR = gamma.*problem.input(:,2);
    Y = [dS; dI; dR];
    params.noNeurons = particle(i).Position(3);
    [particle(i).Cost,particle(i).netWeights] = CostFunction([X;X;X],Y,params);
    
    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    particle(i).Best.netWeights = particle(i).netWeights;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost   
        GlobalBest = particle(i).Best;       
    end
    
end

BestCost = zeros(MaxIt, 1);

%% PSO Main Loop

for it = 1:MaxIt
    
    for i = 1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);
        particle(i).Velocity(nVar) = floor(particle(i).Velocity(nVar));

        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
        IsOutside = (particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);
        
        % Evaluation
        beta = particle(i).Position(1);
        gamma = particle(i).Position(2);
        dS = -beta*problem.input(:,1).*problem.input(:,2)./problem.normalizedN;
        dI = beta*problem.input(:,1).*problem.input(:,2)./problem.normalizedN - gamma.*problem.input(:,2);
        dR = gamma.*problem.input(:,2);
        Y = [dS; dI; dR];        params.noNeurons = particle(i).Position(3);
        [particle(i).Cost,particle(i).netWeights] = CostFunction([X;X;X],Y,params);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            particle(i).Best.netWeights = particle(i).netWeights;

            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost         
                GlobalBest = particle(i).Best;
            end
        end
        
    end
    
    BestCost(it) = GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w = w*wdamp;
    
end

BestSol = GlobalBest;

%% Results
% 
% figure;
% %plot(BestCost, 'LineWidth', 2);
% semilogy(BestCost, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Best Cost');
% grid on;
