%% cost_comparison_2D_SI_vs_CSNewton.m
% Cahn-Hilliard 2D, confronto di costo tra
%   (a) SI   : schema SI-IMEX-RK linearizzato 
%   (b) CS-N : schema convex splitting non lineare (23)
%             risolto con il metodo di Newton
%
clear; clc; close all;

%% Parametri 
L = 1;
N = 512;
h = L/N;
epsCH = 0.02;
Tf = 0.08;

refPower   = 14;        % soluzione di riferimento
costPowers = 7:10;       % passi per il confronto: dt = Tf/2^3, ..., Tf/2^9
refMethod  = 2;         
useDealiasing = true;   
opts.tol   = 1e-9;      % GMRES 
opts.maxit = 300;
newton.tol   = 1e-9;    
newton.maxit = 30;

%% Dato iniziale 
rng(0,'twister');
phi0 = 0.001*rand(N);
phi0 = phi0 - mean(phi0(:));

%% Simboli spettrali 
k = (pi/L)*(0:N-1)';
[Kx,Ky] = ndgrid(k,k);
P.K2 = Kx.^2 + Ky.^2;
P.K4 = P.K2.^2;  P.eps2 = epsCH^2;
P.mask = double(hypot(Kx,Ky)/max(k) <= 2/3 | ~useDealiasing);

%% Schemi
R = {R_CSRK3new()};
names = {'SI-IMEXRK5-3'};
%R = {R_CSRK2(), R_CSRK3new()};
%names = {'SI-IMEXRK3-2','SI-IMEXRK5-3'};

%% Soluzione di riferimento 
refFile = sprintf('CH2D_ref_N%d_eps%g_p%d_3ordine_%d.mat',N,epsCH,refPower,refMethod);
if isfile(refFile)
    load(refFile,'phiRef');
    fprintf('Riferimento caricato da %s\n',refFile);
else
    fprintf('Calcolo riferimento: %s, dt = Tf/2^%d ...\n',names{refMethod},refPower);
    tic
    [phiRef,snaps,tE,E] = solveCH_SI(phi0,Tf,Tf/2^refPower,R{refMethod},P,opts,h);
    fprintf('fatto in %.1f minuti\n',toc/60);
    save(refFile,'phiRef','tE','E');
end

%% Warm-up 
solveCH_SI (phi0,2*Tf/2^costPowers(1),Tf/2^costPowers(1),R{1},P,opts,h);
solveCH_CSN(phi0,2*Tf/2^costPowers(1),Tf/2^costPowers(1),R{1},P,opts,newton,h);

%% Confronto 
costFile = sprintf('CH2D_cost_RK53_N%d_eps%g_p%d-%d.mat',N,epsCH,costPowers(1),costPowers(end));
dts = Tf./2.^costPowers;
nk  = numel(dts);  nm = numel(R);
if isfile(costFile)
    load(costFile);
    fprintf('Risultati caricati da %s\n',costFile);
else
    err_SI = zeros(nm,nk);  err_CS = zeros(nm,nk);
    cpu_SI = zeros(nm,nk);  cpu_CS = zeros(nm,nk);
    lin_SI = zeros(nm,nk);  lin_CS = zeros(nm,nk);   % sistemi lineari risolti
    gm_SI  = zeros(nm,nk);  gm_CS  = zeros(nm,nk);   % iterazioni GMRES totali
    nwt_avg = zeros(nm,nk); nwt_max = zeros(nm,nk);  % iterazioni di Newton per stadio
    dE_SI  = zeros(nm,nk);  dE_CS  = zeros(nm,nk);   % massimo incremento di energia
    nfail  = zeros(nm,nk);                            % stadi in cui Newton non converge
    for m = 1:nm
        for j = 1:nk
            t = tic;
            [phi,~,~,E,st] = solveCH_SI(phi0,Tf,dts(j),R{m},P,opts,h);
            cpu_SI(m,j) = toc(t);
            err_SI(m,j) = norm(phi(:) - phiRef(:))/norm(phiRef(:));
            lin_SI(m,j) = st.nlin;  gm_SI(m,j) = st.ngmres;  dE_SI(m,j) = max(diff(E));
            t = tic;
            [phi,~,~,E,st] = solveCH_CSN(phi0,Tf,dts(j),R{m},P,opts,newton,h);
            cpu_CS(m,j) = toc(t);
            err_CS(m,j) = norm(phi(:) - phiRef(:))/norm(phiRef(:));
            lin_CS(m,j) = st.nlin;  gm_CS(m,j) = st.ngmres;  dE_CS(m,j) = max(diff(E));
            nwt_avg(m,j) = st.nlin/st.nstages;  nwt_max(m,j) = st.maxnewton;
            nfail(m,j) = st.nfail;

            fprintf('%-14s dt = Tf/2^%-2d  CPU SI = %8.1f s   CPU CS-N = %8.1f s\n', ...
                names{m},costPowers(j),cpu_SI(m,j),cpu_CS(m,j));
        end
    end
    save(costFile,'err_SI','err_CS','cpu_SI','cpu_CS','lin_SI','lin_CS', ...
         'gm_SI','gm_CS','nwt_avg','nwt_max','dE_SI','dE_CS','nfail','dts','costPowers');
end

%% Tabelle a schermo
for m = 1:nm
    fprintf('\n==========================================================================================================\n');
    fprintf('  %s : SI linearizzato vs CS non lineare (Newton), N = %d x %d\n',names{m},N,N);
    fprintf('==========================================================================================================\n');
    fprintf('   Nt     dt         err SI      err CS-N   CPU SI[s]  CPU CS-N[s]  speed-up  Newton it/stadio (max)  lin SI  lin CS-N\n');
    for j = 1:nk
        fprintf('%5d  %.3e  %.3e  %.3e  %9.1f  %11.1f  %7.2f   %6.2f (%2d)            %6d  %8d\n', ...
            2^costPowers(j),dts(j),err_SI(m,j),err_CS(m,j),cpu_SI(m,j),cpu_CS(m,j), ...
            cpu_CS(m,j)/cpu_SI(m,j),nwt_avg(m,j),nwt_max(m,j),lin_SI(m,j),lin_CS(m,j));
    end
    fprintf('iterazioni GMRES totali:   SI = %d   CS-N = %d\n',sum(gm_SI(m,:)),sum(gm_CS(m,:)));
    fprintf('massimo incremento di energia:   SI = %.2e   CS-N = %.2e\n',max(dE_SI(m,:)),max(dE_CS(m,:)));
    if any(nfail(m,:)), fprintf('ATTENZIONE: Newton non converge in %d stadi\n',sum(nfail(m,:))); end
end

%% Righe LaTeX 
fprintf('\n%% ---- LaTeX: Nt & err SI & err CS-N & CPU SI & CPU CS-N & speed-up & Newton it./stage \\\\\n');
for m = 1:nm
    fprintf('\\multicolumn{7}{l}{%s} \\\\\n',names{m});
    for j = 1:nk
        fprintf('%d & %s & %s & %.1f & %.1f & %.2f & %.2f \\\\\n',2^costPowers(j), ...
            texnum(err_SI(m,j)),texnum(err_CS(m,j)),cpu_SI(m,j),cpu_CS(m,j), ...
            cpu_CS(m,j)/cpu_SI(m,j),nwt_avg(m,j));
    end
end

%load('CH2D_cost_RK53_N512_eps0.02_p7-10.mat');  
%avgSI = gm_SI./lin_SI;   avgCS = gm_CS./lin_CS;
%for j = 1:numel(costPowers)
%    fprintf('%4d  GMRES it./solve:  SI = %.1f   CS-N = %.1f\n', ...
%        2^costPowers(j), avgSI(1,j), avgCS(1,j));
%end

%% Diagramma errore / tempo CPU
%figure('Color','w','Position',[180 120 720 520]);
%axes('XScale','log','YScale','log');  hold on;  grid on;  box on;
%col = [0.85 0.10 0.10; 0.00 0.35 0.90];
%for m = 1:nm
%    plot(cpu_SI(m,:),err_SI(m,:),'-o', 'Color',col(m,:),'LineWidth',1.2, ...
%         'DisplayName',[names{m} ' (SI)']);
%    plot(cpu_CS(m,:),err_CS(m,:),'--s','Color',col(m,:),'LineWidth',1.2, ...
%         'DisplayName',[names{m} ' (CS-Newton)']);
%end
%xlabel('CPU time [s]','Interpreter','latex');
%ylabel('relative $l_2$-error','Interpreter','latex');
%title(sprintf('$t = %g$, $N = %d^2$',Tf,N),'Interpreter','latex');
%legend('Location','southwest');

%% ============================ Funzioni ============================
function [phi,snaps,tE,E,st] = solveCH_SI(phi,Tf,dt,R,P,opts,h)
% Schema SI linearizzato: un sistema lineare per stadio 
s  = size(R,1);
RE = [zeros(1,s); R(1:end-1,:)];          % RE = S*R
nSteps = round(Tf/dt);
m0 = mean(phi(:));
H = zeros([size(phi) s]);
snaps = [];
E = zeros(nSteps+1,1);  E(1) = energy(phi,P,h);
st = struct('nlin',0,'ngmres',0,'nstages',nSteps*s);
for n = 1:nSteps
    for i = 1:s
        ui = phi;  ue = phi;
        for j = 1:i-1
            ui = ui + dt*R(i,j)*H(:,:,j);
            ue = ue + dt*RE(i,j)*H(:,:,j);
        end
        Dt = dt*R(i,i);
        [v,nit] = solveStage(ui,ue,Dt,P,opts);
        st.nlin = st.nlin + 1;  st.ngmres = st.ngmres + nit;
        H(:,:,i) = (v - ui)/Dt;
    end
    phi = v - mean(v(:)) + m0;
    E(n+1) = energy(phi,P,h);
end
tE = (0:nSteps)'*dt;
end

function [phi,snaps,tE,E,st] = solveCH_CSN(phi,Tf,dt,R,P,opts,newton,h)
% Schema CS non lineare (23):
%   G(v) = v - ui - Dt*Lap( -eps^2 Lap v + v^3 - ue ) = 0
% con il metodo di Newton, partendo da v^0 = ue.
s  = size(R,1);
RE = [zeros(1,s); R(1:end-1,:)];
nSteps = round(Tf/dt);
m0 = mean(phi(:));
H = zeros([size(phi) s]);
snaps = [];
E = zeros(nSteps+1,1);  E(1) = energy(phi,P,h);
st = struct('nlin',0,'ngmres',0,'nstages',nSteps*s,'maxnewton',0,'nfail',0);
for n = 1:nSteps
    for i = 1:s
        ui = phi;  ue = phi;
        for j = 1:i-1
            ui = ui + dt*R(i,j)*H(:,:,j);
            ue = ue + dt*RE(i,j)*H(:,:,j);
        end
        Dt = dt*R(i,i);
        [v,it,nit,conv] = solveStageNewton(ui,ue,Dt,P,opts,newton);
        st.nlin = st.nlin + it;  st.ngmres = st.ngmres + nit;
        st.maxnewton = max(st.maxnewton,it);
        st.nfail = st.nfail + ~conv;
        H(:,:,i) = (v - ui)/Dt;
    end
    phi = v - mean(v(:)) + m0;
    E(n+1) = energy(phi,P,h);
end
tE = (0:nSteps)'*dt;
end

function [v,nit] = solveStage(ui,ue,Dt,P,opts)
% Stadio SI linearizzato 
sz  = size(ui);
c   = 3*ue.^2;
rhs = ui + Dt*idct2(P.K2.*(dct2(ue) + 2*P.mask.*dct2(ue.^3)));
Pk  = 1 + Dt*mean(c(:))*P.K2 + Dt*P.eps2*P.K4;      % precondizionatore spettrale
A   = @(v) reshape(applyA(reshape(v,sz),c,Dt,P),[],1);
M   = @(r) reshape(idct2(dct2(reshape(r,sz))./Pk),[],1);
[v,flag,relres,iter] = gmres(A,rhs(:),[],opts.tol,opts.maxit,M,[],ui(:));
%if flag ~= 0, warning('GMRES: flag = %d, relres = %.1e',flag,relres); end
if flag ~= 0 && relres > 10*opts.tol
    warning('GMRES: flag = %d, relres = %.1e',flag,relres);
end
nit = gmresIters(iter,opts.maxit);
v = reshape(v,sz);
end

function [v,it,nit,conv] = solveStageNewton(ui,ue,Dt,P,opts,newton)
% Stadio CS non lineare risolto con Newton. Lo Jacobiano ha la stessa
% struttura dell'operatore SI, con c = 3v^2 al posto di 3ue^2.
sz   = size(ui);
G    = @(v) v - ui + Dt*idct2(P.K2.*(P.mask.*dct2(v.^3) - dct2(ue)) + P.eps2*P.K4.*dct2(v));
v    = ue;                                   % dato iniziale di Newton
nrm0 = max(norm(ui(:)),1e-14);
it = 0;  nit = 0;  conv = false;
while it < newton.maxit
    r = -G(v);
    if norm(r(:)) <= newton.tol*nrm0, conv = true; break; end
    c  = 3*v.^2;
    Pk = 1 + Dt*mean(c(:))*P.K2 + Dt*P.eps2*P.K4;
    J  = @(w) reshape(applyA(reshape(w,sz),c,Dt,P),[],1);
    M  = @(q) reshape(idct2(dct2(reshape(q,sz))./Pk),[],1);
    [dv,flag,relres,iter] = gmres(J,r(:),[],opts.tol,opts.maxit,M,[],zeros(numel(v),1));
    %if flag ~= 0, warning('GMRES (Newton): flag = %d, relres = %.1e',flag,relres); end
    if flag ~= 0 && relres > 10*opts.tol
        warning('GMRES: flag = %d, relres = %.1e',flag,relres);
    end
    dv  = reshape(dv,sz);
    v   = v + dv;
    it  = it + 1;
    nit = nit + gmresIters(iter,opts.maxit);
    % criterio sull'incremento
    if norm(dv(:)) <= newton.tol*norm(v(:)), conv = true; break; end   
end
    

if ~conv
    r = G(v);  conv = norm(r(:)) <= newton.tol*nrm0;
end
end

function y = applyA(v,c,Dt,P)
% A v = v - Dt*Lap( c.*v - eps^2 Lap v ): 3 trasformate invece di 6
y = v + Dt*idct2(P.K2.*P.mask.*dct2(c.*v) + P.eps2*P.K4.*dct2(v));
end

function n = gmresIters(iter,restart)
% numero totale di iterazioni interne dall'output [outer inner] di gmres
if numel(iter) == 2
    n = (iter(1) - 1)*restart + iter(2);
else
    n = iter;
end
end

function F = energy(phi,P,h)
gx = diff(phi,1,1)/h;  gy = diff(phi,1,2)/h;
F  = h^2*(sum(0.25*(phi(:).^2 - 1).^2) + 0.5*P.eps2*(sum(gx(:).^2) + sum(gy(:).^2)));
end

function s = texnum(x)
% 
e = floor(log10(x));  m = x/10^e;
s = sprintf('$%.3f\\mathrm{e}{%+03d}$',m,e);
end

%% Tableau (parte implicita R; la parte esplicita e' RE = S*R)
function R = R_CSRK2()
R = [2/3 0 0; -7/12 2/3 0; -1/3 2/3 2/3];
end
%
function R = R_CSRK3new()
R = [37020500/137208437 0 0 0 0; 106190902/257663715 31211902/62012697 0 0 0; ...
     -16277267/112129400 -25122194/54005823 44990027/53375095 0 0; ...
     32410717/118168107 -42488/95969011 -27282484/45912445 193798360/227099119 0; ...
     14118261/72107921 -109009254/144718357 13811209/84156809 22662449/35905015 127332368/167066241];
end