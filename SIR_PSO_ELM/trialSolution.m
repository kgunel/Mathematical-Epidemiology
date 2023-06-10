function [y] = trialSolution(X, init, p)
  y = init' + (X )'.*Net(X,p);
  y = y./sum(y,1);
end
