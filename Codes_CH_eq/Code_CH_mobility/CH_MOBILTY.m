clear; 
clc; 
close all;

%% Parameters
T       = 1/8;
epsilon = 1/20;
eps2    = epsilon^2;

% Spatial discretization parameters.
N  = 512;
Nt = 2^8;
dt = T/Nt;

% GMRES parameters
gmresTol = 1e-10;
gmresMaxit = 100;
useDealiasing = true;

assert(mod(N,2)==0, 'N must be even.');
L = 1;
dx = L/N;
x = (-1/2 + (0:N-1)'/N);
k = (2*pi/L)*[0:N/2-1, -N/2:-1]';
k2 = k.^2;
mask = abs(k) <= (2/3)*max(abs(k));
if ~useDealiasing, mask = true(N,1); end

W = tanh(cos(2*pi*x)/(10*epsilon));
u0 = W;

%% 
order = 3;
switch order
    case 2
        [A, At] = SI_IMEXRK3_2_tableau();
        %verifyTableau(A, At);
    case 3
        [A, At] = SI_IMEXRK5_3_tableau();
end


%% Run Algorithm 2
fprintf('Fourier spectral discretization\n');
fprintf('N = %d, Nt = %d, dt = %.6e\n',N,Nt,dt);
fprintf('epsilon = %.6e, eps2 = %.6e\n',epsilon,eps2);


u = u0;
s = size(A,1);
D = @(v) real(ifft(1i*k.*fft(v)));
Dxx = @(v) real(ifft(-k2.*fft(v)));

Energy = zeros(Nt,1);   % preallocazione, opzionale
tic;

for n = 1:Nt
    un = u;
    G = zeros(size(u,1),s);
    for i = 1:s
        UE = un;
        Ubar = un;
        %
        for j = 1:i-1
            UE = UE - dt*At(i,j)*G(:,j);
            Ubar = Ubar - dt*A(i,j)*G(:,j);
        end
        delta = dt*A(i,i);
        muE=(1-epsilon)*(1-UE.^2).^2 + epsilon; %ones(size(UE));
        % 
        UE2 = filter23(UE.^2,mask);
        UE3 = filter23(UE.^3,mask);
        cE = 3*UE2;
        Lmu = @(q) D(filter23(muE.*D(q),mask));
                                                        
        rhs = Ubar - delta*Lmu(2*UE3 + UE - W);
        Afun = @(v) v + delta*Lmu(eps2*Dxx(v)-cE.*v); 

       
        %
        muBar = mean(muE);
        cBar = mean(cE);
        Mfun = @(r) real(ifft(fft(r) ./ ...
            (1 + delta*muBar*(eps2*k2.^2 + cBar*k2))));  

        [UI,flag,relres,iter] = gmres(Afun,rhs,[],gmresTol,gmresMaxit,Mfun,[],Ubar);
    
        if flag ~= 0
           warning(['GMRES did not fully converge: time step %d, ' ...
                'stage %d, flag=%d, relres=%.3e, ' ...
                'iter=[%d %d].'], ...
                n,i,flag,relres,iter(1),iter(2));
        end
       

        UI = real(UI);

        chemical = eps2*Dxx(UI) - cE.*UI + 2*UE3 + UE - W; %FORM 1
        %chemical = eps2*Dxx(UI) - cE.*UI + 2*UE3 + UI - W;  %FORM 2
        G(:,i) = Lmu(chemical);

        
    end
    % A is stiffly accurate 
    u = real(UI);
    u = u - mean(u) + mean(un);       % remove only round-off mass drift

    if mod(n,max(1,round(Nt/10))) == 0
        fprintf(' step %d/%d, min=%+.6e, max=%+.6e, GMRES it=%g\n',...
            n,Nt,min(u),max(u),iter);
    end
    

    ux = real(ifft(1i*k.*fft(u)));

    F  = (u.^2-1).^2/4;
    F_density = 0.5*eps2*(ux.^2) + F + u.*W;       
    Energy(n) = sum(F_density)*dx; 

        %plot(x,u0,'k-','LineWidth',1.5);
        %hold on;
        %plot(x,u,'r','LineWidth',1.5);
        %title(sprintf('Time = %.4f',n*dt));
        %hold off;
        %drawnow
   
end

fprintf('Elapsed time: %.2f s\n',toc);

%% Fig. plot
figure('Color','w');
plot(x,u0,'k-','LineWidth',1.5); hold on;
plot(x,u,'Color','r','LineWidth',1.5);
xlabel('x');
ylabel('u');
xlim([-0.5 0.5]);
ylim([-1.00 1.00]);
box on;
set(gca,'FontSize',12);
legend('initial condition','final solution','Location','best');
title('1D Cahn--Hilliard with mobility');
grid on

% %% ==========================================================
% % ENERGY
% %% ==========================================================
figure
plot((1:Nt)*dt,Energy,'LineWidth',2)

xlabel('t')
ylabel('E(u)')
title('Energia')

grid on


function out = filter23(v,mask)
vhat = fft(v); 
vhat(~mask) = 0; 
out = real(ifft(vhat));
end

function [A, At] = SI_IMEXRK5_3_tableau()
A = [37020500/137208437, 0, 0, 0, 0;
    106190902/257663715, 31211902/62012697, 0, 0, 0;
    -16277267/112129400, -25122194/54005823, 44990027/53375095, 0, 0;
    32410717/118168107, -42488/95969011, -27282484/45912445, 193798360/227099119, 0;
    14118261/72107921, -109009254/144718357, 13811209/84156809, 22662449/35905015, 127332368/167066241];
s = size(A,1);
At = full(spdiags(ones(s,1),-1,s,s)*A);
end

function [A, At] = SI_IMEXRK3_2_tableau()
A = [2/3 0 0;
    -7/12 2/3 0;
    -1/3 2/3 2/3];

%%con tecnica di spostamento
At = [0 0 0;
    2/3 0 0;
    -7/12 2/3 0];
end