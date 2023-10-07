close all
clear
clc

% True prediction
rng(0)
a=20; J=64; x=(0:a/J:a)'; 
% u0=1./(1+exp(-(2-x)/sqrt(2)));
u0 = cos(x);
ell=1; N=500; T=1; epsilon=1; sigma=1;
sample = 200;
sol_true = zeros(J+1,N+1,sample);
for i = 1:sample
    rng(0)
    i
    [t,ut]=spde_fd_n_exp(u0,T,a,N,J,epsilon,sigma,ell,@(u) (0.5*u+0.5*u.^2-u.^3));
    sol_true(:,:,i) = ut;
end

y = sol_true(:,2:end,:)-sol_true(:,1:end-1,:);
dt= T/N;
xdt = (1/dt)*mean(y,3);
xdiff = (1/dt)*mean(y.*y,3);

% Prediction using identified system
rng(0)
eps_arr = normrnd(0.9690,0.0002,200,1);
sig_arr = normrnd(1.0199,0.0002,200,1);
u_arr = normrnd(0.4656,0.0004,200,1);
u2_arr = normrnd(0.5037,0.0001,200,1);
u3_arr = normrnd(0.9979,0.0001,200,1);
sol_pred = zeros(J+1,N+1,sample);
for i = 1:sample
    rng(0)
    i
    [t,ut]=spde_fd_n_exp(u0,T,a,N,J,eps_arr(i),sig_arr(i),ell,@(u) (u_arr(i)*u+u2_arr(i) *u.^2-u3_arr(i)*u.^3));
    sol_pred(:,:,i) = ut;
end

y = sol_pred(:,2:end,:)-sol_pred(:,1:end-1,:);
dt= T/N;
xdt = (1/dt)*mean(y,3);
xdiff = (1/dt)*mean(y.*y,3);
% 
% save('Review_Nagumo_prediction.mat', 'sol_true', 'sol_pred')

%%
figure();
subplot(4,1,1); imagesc(mean(sol_true, 3))
subplot(4,1,2); imagesc(mean(sol_pred, 3))
subplot(4,1,3); imagesc(abs(mean(sol_true, 3) - mean(sol_true, 3)))
subplot(4,1,4); imagesc(std(sol_pred, 3))
