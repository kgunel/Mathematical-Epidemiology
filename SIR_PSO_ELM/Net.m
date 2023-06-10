function y = Net(X,p)
m = floor(length(p)/5);
n = size(X,1);
alpha = p(1:m);
w = p(m+1:2*m);
omega = p(2*m+1:3*m);
mu = p(3*m+1:4*m);
bias = p(4*m+1:5*m);
bias2 = p(5*m+1);

y = zeros(1,n);
z = zeros(m,n);
for j=1:n
    for i=1:m
        z(i,j) = w(i)*X(j,1)+omega(i)*X(j,2)+ mu(i)*X(j,3) + bias(i);
        y(j) = y(j) + alpha(i)*sigma(z(i,j)) + bias2;
    end
end