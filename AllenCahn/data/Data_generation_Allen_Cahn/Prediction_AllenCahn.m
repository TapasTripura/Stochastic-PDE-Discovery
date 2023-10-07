close all
clear
clc
rng(0)

% True prediction
rng(0)
a=20; J=64; x=(0:a/J:a)'; 
u0 = sin(x);
ell=1; N=400; T=1; epsilon=1; sigma=1;
sample =200;
sol_true = zeros(J+1,N+1,sample);
for i = 1:sample
    rng(0)
    i
    [t,ut]=spde_fd_n_exp(u0,T,a,N,J,epsilon,sigma,ell,@(u) (u-u.^3));
    sol_true(:,:,i) = ut;
end
y = sol_true(:,2:end,:)-sol_true(:,1:end-1,:);
dt= T/N;
xdt = (1/dt)*mean(y,3);
xdiff = (1/dt)*mean(y.*y,3);

% Prediction using identified system
eps_arr = normrnd(0.99317,0.00014,200,1);
sig_arr = normrnd(0.99,3.77e-8,200,1);
u_arr = normrnd(1.0394,0.0012,200,1);
u3_arr = normrnd(1.0392,0.0013,200,1);
sol_pred = zeros(J+1,N+1,sample);
for i = 1:sample
    rng(0)
    i
    [t,ut]=spde_fd_n_exp(u0,T,a,N,J,eps_arr(i),sig_arr(i),ell,@(u) (u_arr(i)*u-u3_arr(i)* u.^3));
    sol_pred(:,:,i) = ut;
end
y = sol_pred(:,2:end,:)-sol_pred(:,1:end-1,:);
dt= T/N;
xdt = (1/dt)*mean(y,3);
xdiff = (1/dt)*mean(y.*y,3);

save('Review_Allen_cahn_prediction.mat', 'sol_true', 'sol_pred')

%%
figure();
subplot(4,1,1); imagesc(mean(sol_true, 3))
subplot(4,1,2); imagesc(mean(sol_pred, 3))
subplot(4,1,3); imagesc(abs(mean(sol_true, 3) - mean(sol_true, 3)))
subplot(4,1,4); imagesc(std(sol_pred, 3))
