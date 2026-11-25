%   Cahn-Hilliard 1D con mobilità, Accuracy
%      
clear; clc; close all;

%% Parametri
%% 
ep   = 0.3;
T    = 1/8;
xmin = -0.5;  xmax = 0.5;  L = xmax - xmin;
N    = 128;
Nt_list = 2.^(6:11);

W  = @(x) tanh(cos(2*pi*x) / (10*ep));
mu =  @(u) (1-ep)*(1-u.^2).^2 + ep; 

%%  (FFT)
dx = L/N;
x  = xmin + (0:N-1)'*dx;
k  = (2*pi/L) * [0:N/2-1, -N/2:-1]';              
k1 = k;  k1(N/2+1) = 0;                          
d1 = @(v) real(ifft(1i*k1   .* fft(v)));          
d2 = @(v) real(ifft(-(k.^2) .* fft(v)));          
Wv = W(x);

method = 1;
switch method
    case 0
        A = [2/3 0 0;
    -7/12 2/3 0;
    -1/3 2/3 2/3];
    case 1
%% Tableau SI-IMEXRK5-3
A = [ 37020500/137208437,                   0,                  0,                   0,                   0;
     106190902/257663715,  31211902/62012697,                  0,                   0,                   0;
     -16277267/112129400, -25122194/54005823, 44990027/53375095,                   0,                   0;
      32410717/118168107,    -42488/95969011, -27282484/45912445, 193798360/227099119,                   0;
      14118261/72107921, -109009254/144718357, 13811209/84156809,  22662449/35905015, 127332368/167066241];
end

s  = size(A,1);
At = [zeros(1,s); A(1:s-1,:)];                    % Atilde = S*A

%% 
sols = cell(numel(Nt_list),1);
for m = 1:numel(Nt_list)
    Nt = Nt_list(m);  dt = T/Nt;
    u  = Wv;     % u(x,0) = W(x)
    tic
    for n = 1:Nt
        u = step_imexrk(u, dt, A, At, d1, d2, k, ep, Wv, mu);
    end
    sols{m} = u;
    fprintf('Nt = %4d  dt = %.5f,  errore massa = %.2e\n', ...
            Nt, dt, abs(dx*sum(u) - dx*sum(Wv)));
end

%% Error and order
dt_list = T ./ Nt_list(1:end-1);
L2   = zeros(1,numel(dt_list));
Linf = zeros(1,numel(dt_list));
for m = 1:numel(dt_list)
    d = sols{m} - sols{m+1};
    L2(m)   = sqrt(dx*sum(d.^2));
    Linf(m) = max(abs(d));
end
rate_L2   = log2(L2(1:end-1)   ./ L2(2:end));
rate_Linf = log2(Linf(1:end-1) ./ Linf(2:end));

fprintf('\n  dt          L2-diff       ordine    Linf-diff     ordine\n');
for m = 1:numel(dt_list)
    if m == 1
        fprintf('  %.5f   %.4e     ---       %.4e     ---\n', dt_list(m), L2(m), Linf(m));
    else
        fprintf('  %.5f   %.4e   %5.2f     %.4e   %5.2f\n', dt_list(m), L2(m), rate_L2(m-1), Linf(m), rate_Linf(m-1));
    end
end

%% ========================================================================
function u = step_imexrk(u, dt, A, At, d1, d2, k, ep, Wv, mu)
% 
    s  = size(A,1);
    G  = zeros(numel(u), s);                          
    e2 = ep^2;

    for i = 1:s
        UE = u - dt * G(:,1:i-1) * At(i,1:i-1)';     
        Ut = u - dt * G(:,1:i-1) * A(i,1:i-1)';       
        m  = mu(UE);
        q  = 3*UE.^2; 
        Dt  = dt * A(i,i);
        LM = @(v) d1(m .* d1(v));                     
        %linearized system:  
        Afun = @(v) v + Dt * LM(e2*d2(v) - q.*v);
        rhs  = Ut - Dt * LM(2*UE.^3 + UE - Wv);
        % Precondicioner in Fourier 
        simbolo = 1 + Dt * mean(m) * k.^2 .* (e2*k.^2 + max(mean(q), 0));
        Pinv    = @(v) real(ifft(fft(v) ./ simbolo));
        [UI, flag, relres, iter]  = gmres(Afun, rhs, [], 1e-13, 100, Pinv, [], UE);
        % 
        G(:,i) = LM(e2*d2(UI) - q.*UI + 2*UE.^3 + UE  - Wv);
    end
    % stiffly accurate: u^{n+1} = U_s^I
    u = UI;                                           
end