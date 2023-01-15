% The upper-triangular elements the matrix P is converted into a row
r=P1;
% r=p2;
% r=p3;
  N=length(r);
a=zeros([1,N*(N-1)/2]);
  for w=1:N
    l=(w-1)*(2*N-w)/2+1;
    a1(1,l:(l+N-w-1))=r((w+1):end,w);
  end 
%merge
x=[a1;a2;a3];

% The weight vector by the ReliefF algorithm rx;

%Restore to the symmetric matrix
x_1=zeros(62,62);
for i=1:62
    x_1(i,i+1:62)=rx(62*(i-1)-i*(i-1)/2+1:62*i-i*(i+1)/2);
end
for i=1:62
x_1(i+1:62,i)=rx(62*(i-1)-i*(i-1)/2+1:62*i-i*(i+1)/2);
end

% binarized
A=x_1;
t=0.07;
A(A<t)=0;
A(A>=t)=1;

