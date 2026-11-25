%% Cahn-Hilliard 2D, condizioni di Neumann (DCT), schemi CSRK semi-impliciti
%  - snapshot e energia della soluzione di riferimento
%  - errore l2 relativo e ordine di accuratezza a t = Tf
clear; clc; close all;
%
%% Parametri
L = 1;  
N = 512;  
h = L/N;  
x = ((1:N) - 0.5)*h;
epsCH = 0.02;  
Tf = 0.08;
tSnap = [0.01 0.02 0.04 0.08];

refPower   = 14;        % riferimento: dt = Tf/2^14
testPowers = 3:12;      % test:       dt = Tf/2^3, ..., Tf/2^12
refMethod  = 2;         % 1. second order 2. third order  
runConvergence = true;   % false: solo snapshot ed energia
useDealiasing  = true;  % filtro 2/3 sui termini nonlineari

opts.tol = 1e-9; 
opts.maxit = 300;    % GMRES

%% Dato iniziale (lo stesso per riferimento e test)
rng(0,'twister');
phi0 = 0.001*rand(N);  
phi0 = phi0 - mean(phi0(:));

%% Simboli spettrali (DCT-II -> Neumann omogenee)
k = (pi/L)*(0:N-1)';  
[Kx,Ky] = ndgrid(k,k);
P.K2 = Kx.^2 + Ky.^2;  
P.K4 = P.K2.^2;  P.eps2 = epsCH^2;
P.mask = double(hypot(Kx,Ky)/max(k) <= 2/3 | ~useDealiasing);

%% Schemi
R = {R_CSRK2(), R_CSRK3new()};
names = {'CSRK-R(2)','CSRK-R(3) NEW'};
order = [2 3];

%% Soluzione di riferimento (calcolata una volta e salvata)
refFile = sprintf('CH2D_ref_N%d_eps%g_p%d_3ordine_%d.mat',N,epsCH,refPower,refMethod);
if isfile(refFile)
    load(refFile,'phiRef','snaps','tE','E');
    fprintf('Riferimento caricato da %s\n',refFile);
else
    disp('ciao')
    fprintf('Calcolo riferimento: %s, dt = Tf/2^%d ...\n',names{refMethod},refPower);
    tic
    [phiRef,snaps,tE,E] = solveCH(phi0,Tf,Tf/2^refPower,R{refMethod},P,opts,tSnap,h);
    fprintf('fatto in %.1f minuti\n',toc/60);
    save(refFile,'phiRef','snaps','tE','E');
end

%% Snapshot
figure('Color','w','Position',[100 150 1120 290]);
tiledlayout(1,numel(tSnap),'TileSpacing','compact','Padding','compact');
for j = 1:numel(tSnap)
    nexttile;
    imagesc(x,x,snaps(:,:,j));  axis image off;  set(gca,'YDir','normal');
    colormap(gca,jet(256));  clim([-1.1 1.1]);    % caxis nelle versioni < R2022a
    title(sprintf('$t = %g$',tSnap(j)),'Interpreter','latex');
end

%% Energia libera
figure('Color','w');
plot(tE,E,'LineWidth',1.5,'Color',[0.85 0.10 0.10]);  grid on;
xlabel('$t$','Interpreter','latex');  ylabel('$F(u)$','Interpreter','latex');
title('CH 2D: energia libera','Interpreter','latex');

if ~runConvergence, return; end

%% Studio di convergenza a t = Tf (errori salvati su file)
dts = Tf./2.^testPowers;
errFile = sprintf('CH2D_err_N%d_eps%g_p%d.mat',N,epsCH,refPower);
if isfile(errFile)
    load(errFile,'err');
else
    err = nan(numel(R),numel(dts));
    for m = 1:numel(R)
        for j = 1:numel(dts)
            phi = solveCH(phi0,Tf,dts(j),R{m},P,opts,[],h);
            err(m,j) = norm(phi(:) - phiRef(:))/norm(phiRef(:));
            fprintf('%-20s dt = Tf/2^%-2d  errore = %.3e\n',names{m},testPowers(j),err(m,j));
        end
    end
    save(errFile,'err','dts');
end

% Ordine osservato tra passi consecutivi (dt dimezzato)
for m = 1:numel(R)
    fprintf('%-20s ordini: %s\n',names{m},sprintf('%5.2f ',log2(err(m,1:end-1)./err(m,2:end))));
end

%% Grafico di convergenza
figure('Color','w','Position',[180 120 720 520]);
axes('XScale','log','YScale','log');  hold on;  grid on;  box on;
col = [0.85 0.10 0.10; 0.00 0.35 0.90; 0 0 0];  mk = {'*','o'};
for m = 1:numel(R)
    plot(dts,err(m,:),['-' mk{m}],'Color',col(m,:),'LineWidth',1.2, ...
         'DisplayName',names{m});
end
tri(dts,err(1,:),numel(dts)-2,order(1));     % triangolo di pendenza 2
tri(dts,err(2,:),numel(dts)-2,order(2));     % triangolo di pendenza 3
xlabel('time step $\Delta t$','Interpreter','latex');
ylabel('relative $l_2$-error','Interpreter','latex');
title(sprintf('$t = %g$',Tf),'Interpreter','latex');
legend('Location','southeast');

%% ============================ Funzioni ============================
function [phi,snaps,tE,E] = solveCH(phi,Tf,dt,R,P,opts,tSnap,h)
% Schema CSRK semi-implicito
s  = size(R,1);
RE = [zeros(1,s); R(1:end-1,:)];          % RE = S*R
nSteps = round(Tf/dt);
m0 = mean(phi(:));
H = zeros([size(phi) s]);
snaps = zeros([size(phi) numel(tSnap)]);
E = zeros(nSteps+1,1);  E(1) = energy(phi,P,h);
for n = 1:nSteps
    for i = 1:s
        ui = phi;  ue = phi;
        for j = 1:i-1
            ui = ui + dt*R(i,j)*H(:,:,j);
            ue = ue + dt*RE(i,j)*H(:,:,j);
        end
        Dt = dt*R(i,i);
        v  = solveStage(ui,ue,Dt,P,opts);
        H(:,:,i) = (v - ui)/Dt;
    end
    phi = v - mean(v(:)) + m0;            
    E(n+1) = energy(phi,P,h);
    j = find(abs(n*dt - tSnap) < dt/2);
    if ~isempty(j), snaps(:,:,j) = phi; end
end
tE = (0:nSteps)'*dt;
end

function v = solveStage(ui,ue,Dt,P,opts)
sz  = size(ui);
c   = 3*ue.^2;
rhs = ui + Dt*idct2(P.K2.*(dct2(ue) + 2*P.mask.*dct2(ue.^3)));
Pk  = 1 + Dt*mean(c(:))*P.K2 + Dt*P.eps2*P.K4;      % precondizionatore spettrale
A   = @(v) reshape(applyA(reshape(v,sz),c,Dt,P),[],1);
M   = @(r) reshape(idct2(dct2(reshape(r,sz))./Pk),[],1);
[v,flag,relres] = gmres(A,rhs(:),[],opts.tol,opts.maxit,M,[],ui(:));
if flag ~= 0, warning('GMRES: flag = %d, relres = %.1e',flag,relres); end
v = reshape(v,sz);
end

function y = applyA(v,c,Dt,P)
% A v = v - Dt*Lap( c.*v - eps^2 Lap v ): 3 trasformate invece di 6
y = v + Dt*idct2(P.K2.*P.mask.*dct2(c.*v) + P.eps2*P.K4.*dct2(v));
end

function F = energy(phi,P,h)
gx = diff(phi,1,1)/h;  gy = diff(phi,1,2)/h;
F  = h^2*(sum(0.25*(phi(:).^2 - 1).^2) + 0.5*P.eps2*(sum(gx(:).^2) + sum(gy(:).^2)));
end

function tri(dts,e,i,p)
% triangolo di pendenza p, sotto la curva e(dts) vicino al punto i
x1 = dts(i)*1.15;  y1 = e(i)/4;  x2 = 2*x1;  y2 = y1*2^p;
plot([x1 x2],[y1 y1],'k',[x2 x2],[y1 y2],'k','LineWidth',1.2,'HandleVisibility','off');
plot([x1 x2],[y1 y2],'--k','HandleVisibility','off');
text(x2*1.1,sqrt(y1*y2),num2str(p),'HandleVisibility','off');
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