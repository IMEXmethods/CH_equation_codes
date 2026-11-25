%% Cahn-Hilliard 1D con schemi semi-impliciti SI-IMEXRK2 e SI-IMEXRK5-3
%
clear; clc; close all;

%% Parametri
Nx = 128; 
L = 1; 
dx = L/Nx; 
Tf = 0.08;
k  = 2*pi/L*[0:Nx/2-1, -Nx/2:-1]';            
p  = struct('ep',0.02, 'k',k, 'k2',k.^2, 'dx',dx, 'Nx',Nx);

rng(0);  
u0 = 0.01*(2*rand(Nx,1) - 1);        % dato iniziale casuale

kE   = [3 5 7];        %  dt = Tf/2^kE
kk   = 3:14;           %  dt = Tf/2^kk
kref = 18;             % reference solution dt = Tf/2^kref 

%% Tableau
%Implicit part of the Second order SI-IMEXRK2 method
A2 = [ 2/3    0   0;
      -7/12  2/3  0;
      -1/3   2/3  2/3 ];

%Implicit part of the Third order SI-IMEXRK5-3 method

A3 = [  37020500/137208437            0                      0                      0                      0;
       106190902/257663715     31211902/62012697             0                      0                      0;
       -16277267/112129400    -25122194/54005823      44990027/53375095             0                      0;
        32410717/118168107        -42488/95969011    -27282484/45912445     193798360/227099119           0;
        14118261/72107921    -109009254/144718357     13811209/84156809      22662449/35905015     127332368/167066241 ];

M = {'SI-IMEXRK2', A2;  'SI-IMEXRK5-3', A3};

%% Reference solution (S-IMEXRK5-3) 
refFile = 'ref_sol.mat';
par     = [Nx, Tf, p.ep, kref];                
Yes = true;
if isfile(refFile)
    R  = load(refFile);
    Yes = isequal(R.par, par) && isequal(R.u0, u0);
end

if Yes
    uref = R.uref;  Eref = R.Eref;
    fprintf('Riferimento caricato da %s\n', refFile);
else
    fprintf('Calcolo il riferimento (dt = Tf/2^%d)...\n', kref);
    [uref, Eref] = ch_solve(A3, u0, Tf, 2^kref, p);
    save(refFile, 'uref', 'Eref', 'par', 'u0');
end
tref = linspace(0, Tf, 2^kref + 1);

% 
%% Figure 1 e 2:  Energy Evolution
col = {[0.4 0.7 0.2], 'b', 'r'};
for m = 1:2
    figure(m); clf; hold on
    for j = 1:numel(kE)
        N = 2^kE(j);
        [~, E] = ch_solve(M{m,2}, u0, Tf, N, p);
        plot(linspace(0,Tf,N+1), E, '--', 'Color', col{j}, 'LineWidth', 1.2, ...
             'DisplayName', sprintf('$\\Delta t = T_f/2^{%d}$', kE(j)));
    end
    plot(tref, Eref, 'k-', 'LineWidth', 1.4, 'DisplayName', 'Reference');
    title(M{m,1}, 'Interpreter', 'latex','FontSize', 24);
    xlabel('time', 'Interpreter', 'latex','FontSize', 24);
    ylabel('Energy Evolution', 'Interpreter', 'latex','FontSize', 24);
    axis([0 Tf 0.14 0.26]); grid on;
    legend('Location', 'northeast', 'Interpreter', 'latex','FontSize', 14);
end

%% Relative errors l2 
dts = Tf./2.^kk;   err = zeros(numel(kk), 2);
for m = 1:2
    for j = 1:numel(kk)
        u = ch_solve(M{m,2}, u0, Tf, 2^kk(j), p);
        err(j,m) = norm(u - uref)/norm(uref);
    end
end

figure(3)
h = loglog(dts, err(:,1), 'r-o', dts, err(:,2), 'b-o', 'LineWidth', 1.2); hold on
tri(dts, err(:,1), 11, 2);   % (2)
tri(dts, err(:,2), 11, 3);   % (3)
grid on;
xlabel('time step $\Delta t$', 'Interpreter', 'latex','FontSize', 20);
ylabel('Relative $l_2$-Error', 'Interpreter', 'latex','FontSize', 20);
legend(h, M{1,1}, M{2,1}, 'Location', 'southeast','FontSize', 14);
 

%% ------------------------------------------------------------------------
function [u, E] = ch_solve(A, u, Tf, N, p)
% 
s   = size(A,1);  dt = Tf/N;
At  = [zeros(1,s); A(1:end-1,:)];  % MATRIX: At = S*A
tol = min(1e-10, dt^2);
lap = @(v) real(ifft(-p.k2 .* fft(v)));
E = zeros(1, N+1);  
E(1) = energy(u, p);
for n = 1:N
    H = zeros(p.Nx, s); 
    for i = 1:s
        ue = u + dt * H(:,1:i-1) * At(i,1:i-1)';   % Explicit Step
        ui = u + dt * H(:,1:i-1) * A(i,1:i-1)';    % Implicit Step
        Dt = A(i,i) * dt;
        c  = 3*ue.^2;
        rhs  = ui - Dt * lap(ue + 2*ue.^3);
        Lin  = @(v) v - Dt * lap(c.*v - p.ep^2 * lap(v));
        Pk   = 1 + Dt * (mean(c)*p.k2 + p.ep^2 * p.k2.^2);
        Pinv = @(r) real(ifft(fft(r) ./ Pk));
        [v, flag, relres, iter] = gmres(Lin, rhs, [], tol, p.Nx, Pinv, [], ui);
        H(:,i) = (v - ui) / Dt;
    end
    % stiffly accurate
    u = v;                                      
    E(n+1) = energy(u, p);
end
end

function F = energy(u, p)
% F[u] = int( ep^2/2 u_x^2 + (u^2-1)^2/4 ) dx
ux = real(ifft(1i * p.k .* fft(u)));
F  = p.dx * sum(0.5*p.ep^2*ux.^2 + 0.25*(u.^2 - 1).^2);
end

function tri(dts, e, i, sl)
% slope sl (log-log)
x1 = dts(i)*1.4;   y1 = e(i);
x2 = x1*2;         y2 = y1*2^sl;        
plot([x1 x2], [y1 y1], 'k',  'LineWidth', 1.5, 'HandleVisibility', 'off')
plot([x2 x2], [y1 y2], 'k',  'LineWidth', 1.5, 'HandleVisibility', 'off')
plot([x1 x2], [y1 y2], '--k','LineWidth', 1,   'HandleVisibility', 'off')
text(x2*0.8, sqrt(y1*y2), num2str(sl), 'HandleVisibility', 'off')
end
%1.1

