function displayResults(x,y1,y2,y3,y1_t,y2_t,y3_t,caption,label,params)
 n = length(x);
  MSE_S = sum((y1-y1_t).^2)/n;
  MSE_I = sum((y2-y2_t).^2)/n;
  MSE_R = sum((y3-y3_t).^2)/n;
  
  % N = params.nPop; % # of individuals 
  m = params.noNeurons; % # of neurons 
  Low = params.VarMin; % lower bound of search space  
  Up = params.VarMax;  % upper bound of search space
  % Dim = params.dim; % dimension of search space
  % max_it = params.Max_Iteration;  % Maximum number of iteration
  algo = params.algo;
  
  fileName='Results\table.tex';
  
  fileID=fopen(fileName,'w+');
  fprintf(fileID,'\\begin{table}[ht]\n');
  fprintf(fileID,['  \\caption{Parameters for ' algo '}\n']);
  fprintf(fileID,'  \\centering\n');
  fprintf(fileID,'  \\begin{tabular}{rr}\n');
  fprintf(fileID,'    \\hline\\hline\n');
  fprintf(fileID,'    %s & %s \n','Parameter','Value\\');
  %fprintf(fileID,'    %s & %d\\ \n','Number of Agents',N);
  fprintf(fileID,'    %s & %d\\ \n','Number of neurons',m);
  fprintf(fileID,'    %s & [%d, %d]\\ \n','Range of search space',Low,Up);
  %fprintf(fileID,'    %s & %d\\ \n','Dimension of search space',Dim);
  %fprintf(fileID,'    %s & %d\\ \n','Maximum Iteration',max_it);
  fprintf(fileID,'    \\hline\n');
  fprintf(fileID,'  \\end{tabular}\n');
  fprintf(fileID,'  \\label{%s}\n',label);
  fprintf(fileID,'\\end{table}\n');  

  fprintf(fileID,'\\begin{figure}\n');
  fprintf(fileID,'  \\centering\n');
  fprintf(fileID,['  \\includegraphics[width=7cm]{figs\\Figure_' algo '_trainSet.eps}\n'] );
  fprintf(fileID,['  \\caption{Numerical solution obtained via ' algo 'for training set}\\label{fig:' algo '_trainSet}\n']);
  fprintf(fileID,'\\end{figure}\n');

  fprintf(fileID,'\\begin{figure}\n');
  fprintf(fileID,'  \\centering\n');
  fprintf(fileID,['  \\includegraphics[width=7cm]{figs\\Figure_' algo '_testSet.eps}\n']);
  fprintf(fileID,['  \\caption{Numerical solution obtained via ' algo 'for test set}\\label{fig:' algo '_testSet}\n']);
  fprintf(fileID,'\\end{figure}\n');
  
  fprintf(fileID,'\\begin{table}[ht]\n');
  fprintf(fileID,'  \\caption{%s}\n',caption);
  fprintf(fileID,'  \\centering\n');
  fprintf(fileID,'  \\begin{tabular}{rccccccc}\n');
  fprintf(fileID,'    %s & %s & %s & %s & %s & %s & %s & %s\n','$k$','$x_{k}$','$S(k)$','$I(k)$','$R(k)$','$E_S$','$E_I$','$E_R$\\');
  fprintf(fileID,'    \\hline\\hline\n');
  fprintf('\n k  x(k)   S(k)      I(k)      R(k)      S_test(k) I_test(k) R_test(k) E_S       E_T        E_R\n');
  fprintf('-----------------------------------------------------------------------------------------------------------------\n');
  for k=1:n
      fprintf('%3d %3d %12.3e %6.3e %6.3e %6.3e %6.3e %6.3e %9.3e %9.3e  %9.3e \n', k, x(k), y1(k), y2(k), y3(k), y1_t(k), y2_t(k), y3_t(k), abs(y1(k)-y1_t(k)), abs(y2(k)-y2_t(k)), abs(y3(k)-y3_t(k)));
      fprintf(fileID,'    %3d & %3d & %12.3f & %9.3f & %9.3e & %9.3e & %9.3e &  %9.3e \\\\ \n', k, x(k), y1_t(k), y2_t(k), y3_t(k), abs(y1(k)-y1_t(k)), abs(y2(k)-y2_t(k)), abs(y3(k)-y3_t(k)));
  end
  fprintf('-----------------------------------------------------------------------------------------------------------------\n\n');
  fprintf('Mean Squared Error for S : %1.3e\n',MSE_S);
  fprintf('Mean Squared Error for I : %1.3e\n',MSE_I);
  fprintf('Mean Squared Error for R : %1.3e\n',MSE_R);
  fprintf(fileID,'    \\hline\n');
  fprintf(fileID,'     \\multicolumn{6}{l}{\\textbf{Mean Squared Error for S :} %1.3e}\\\\ \n',MSE_S);
  fprintf(fileID,'     \\multicolumn{6}{l}{\\textbf{Mean Squared Error for I:} %1.3e}\\\\ \n',MSE_I);
  fprintf(fileID,'     \\multicolumn{6}{l}{\\textbf{Mean Squared Error for R:} %1.3e}\\\\ \n',MSE_R);

  fprintf(fileID,'  \\end{tabular}\n');
  fprintf(fileID,'  \\label{%s}\n',label);
  fprintf(fileID,'\\end{table}\n');  

  fclose(fileID); 
end