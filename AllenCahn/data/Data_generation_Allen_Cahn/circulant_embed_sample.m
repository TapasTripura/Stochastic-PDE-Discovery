function [X,Y]=circulant_embed_sample(c)
tilde_c=[c; c(end-1:-1:2)];    %create first column of C_tilde
[X,Y]=circ_cov_sample(tilde_c); %obtain 2 samples from N(0,C_tilde)
N=length(c); X=X(1:N); Y=Y(1:N); %extract samples from N(0,C)
    
    
    
