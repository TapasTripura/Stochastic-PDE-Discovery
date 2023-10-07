close all
clear
clc

% See An Introduction to Computational Stochastic PDEs 
% - by Gabriel J. Lord, Catherine E. Powell, Tony Shardlow

rng(0)

% Domain:
a=20; 
J=64;
x=(0:a/J:a)'; 

% Initial condition:
u0 = sin(x);

% System parameters:
ell=1; N=500; T=1; epsilon=1; sigma=1;

sample =2000;
sol = zeros(J+1,N+1,sample);
% Time integration:
for i = 1:sample
    i
    [t,ut]=spde_fd_n_exp(u0,T,a,N,J,epsilon,sigma,ell,@(u) 0);
    sol(:,:,i) = ut;
end

% Linear variation:
y = sol(:,2:end,:)-sol(:,1:end-1,:);
dt= T/N;

% Extended Kramers-Moyal moments:
xdt = (1/dt)*mean(y,3);
xdiff = (1/dt)*mean(y.*y,3);

% save('Review_heat_dx_64m_500t.mat', 'xdt', 'xdiff', 'sol')

%%
figure();
surf(mean(sol, 3))
