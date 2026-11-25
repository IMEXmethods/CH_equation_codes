%% search_shifted_imex_rk3_spd.m
% Search for a lower-triangular DIRK matrix R such that:
%
%   Atilde = S*R
%   Rtilde = R - S*R
%
% where S(i,j)=delta_{i,j+1}, and:
%   1) the implicit part is stiffly accurate: b_i = R(s,i);
%   2) the explicit part uses the same weights: btilde_i = b_i;
%   3) the coupled IMEX RK order conditions up to order 3 are satisfied;
%   4) symmetric part of Rtilde is positive definite.
%
% Requires Optimization Toolbox for fmincon.

clear; clc;

%% Choose number of stages
s = 5;              % try s = 4 NO or s = 5 YES!!!!
numStarts = 100;    % increase if no feasible matrix is found
spdMargin = 1e-12;  %1e-10; % min eig((Rtilde+Rtilde')/2) >= spdMargin
targetResidual = 1e-12;

rng(4);

%% Optimization setup
nvar = s*(s+1)/2;   % lower triangular entries of R

optsSearch = optimoptions("fmincon", ...
    Algorithm="sqp", ...
    Display="iter", ...
    MaxFunctionEvaluations=1e6, ...
    MaxIterations=10000, ...
    OptimalityTolerance=1e-14, ...
    ConstraintTolerance=1e-14, ...
    StepTolerance=1e-15);

optsRefineSQP = optimoptions("fmincon", ...
    Algorithm="sqp", ...
    Display="iter", ...
    MaxFunctionEvaluations=1e6, ...
    MaxIterations=10000, ...
    OptimalityTolerance=1e-14, ...
    ConstraintTolerance=1e-14, ...
    StepTolerance=1e-15);

optsRefineInterior = optimoptions("fmincon", ...
    Algorithm="interior-point", ...
    Display="iter", ...
    MaxFunctionEvaluations=1e6, ...
    MaxIterations=10000, ...
    OptimalityTolerance=1e-14, ...
    ConstraintTolerance=1e-14, ...
    StepTolerance=1e-15);

best = [];
bestScore = inf;

for startId = 1:numStarts
    z0 = initialGuess(s, startId);

    obj = @(z) objective(z, s);
    nonlcon = @(z) constraints(z, s, spdMargin);

    try
        [z, fval, exitflag] = fmincon(obj, z0, [], [], [], [], [], [], nonlcon, optsSearch);
    catch ME
        warning("Start %d failed: %s", startId, ME.message);
        continue;
    end

    R = unpackLowerTriangular(z, s);
    info = check_shifted_imex_rk3(R, targetResidual);

    score = max([ ...
        info.normResidualIMEX, ...
        max(0, spdMargin - info.minEigSymRtilde)]);

    fprintf("start=%d, fval=%.3e, exitflag=%d, feasibility score=%.3e\n\n", ...
        startId, fval, exitflag, score);

    if score < bestScore
        bestScore = score;
        best.R = R;
        best.info = info;
        best.exitflag = exitflag;
        best.fval = fval;
    end

    if score < targetResidual
        fprintf("Feasible candidate found.\n");
        break;
    end
end

if isempty(best)
    error("No optimization run completed.");
end

R = best.R;

%% Local refinement from the best candidate
fprintf("\nRefining best candidate with SQP...\n");
[R, info, fvalRefine1, exitflagRefine1] = refineCandidate( ...
    R, s, spdMargin, optsRefineSQP, targetResidual);

fprintf("\nRefining best candidate with interior-point...\n");
[R, info, fvalRefine2, exitflagRefine2] = refineCandidate( ...
    R, s, spdMargin, optsRefineInterior, targetResidual);

format long
save('R_SIIMEXRK53.mat','R');

fprintf("\nBest candidate for s=%d:\n", s);
disp("R =");       disp(R);
%fprintf('%23.16e %23.16e %23.16e %23.16e %23.16e\n', R');
disp("Atilde =");  disp(info.Atilde);
disp("Rtilde =");  disp(info.Rtilde);
disp("eig symmetric Rtilde ="); disp(info.eigSymRtilde.');
fprintf("Final IMEX residual: %.16e\n", info.normResidualIMEX);
fprintf("Final min eig symmetric Rtilde: %.16e\n", info.minEigSymRtilde);
fprintf("Refinement SQP: fval=%.3e, exitflag=%d\n", fvalRefine1, exitflagRefine1);
fprintf("Refinement interior-point: fval=%.3e, exitflag=%d\n", fvalRefine2, exitflagRefine2);




%% Local functions
function [R, info, fval, exitflag] = refineCandidate(R0, s, spdMargin, opts, tol)
    z0 = packLowerTriangular(R0);
    obj = @(z) objective(z, s);
    nonlcon = @(z) constraints(z, s, spdMargin);

    [z, fval, exitflag] = fmincon(obj, z0, [], [], [], [], [], [], nonlcon, opts);

    R = unpackLowerTriangular(z, s);
    info = check_shifted_imex_rk3(R, tol);
end

function f = objective(z, s)
    R = unpackLowerTriangular(z, s);
    [Atilde, Rtilde] = shiftedMatrices(R);

    bR = R(end,:).';

    res = imexRK3ResidualCommonWeights(R, Atilde, bR);

    symPart = 0.5*(Rtilde + Rtilde.');
    minEig = min(eig(symPart));

    % Keep the search numerically tame; order residual dominates near feasible points.
    f = sum(res.^2) + 1e-8*sum(z.^2) + 1e4*max(0, -minEig)^2;
end

function [c, ceq] = constraints(z, s, spdMargin)
    R = unpackLowerTriangular(z, s);
    [Atilde, Rtilde] = shiftedMatrices(R);

    bR = R(end,:).';

    ceq = imexRK3ResidualCommonWeights(R, Atilde, bR);

    symPart = 0.5*(Rtilde + Rtilde.');
    eigSym = eig(symPart);

    % fmincon enforces c <= 0.
    c = spdMargin - eigSym;
end

function z0 = initialGuess(s, startId)
    gamma = 0.5;
    R0 = zeros(s);

    for i = 1:s
        R0(i,i) = gamma;
        for j = 1:i-1
            R0(i,j) = 0.15*randn();
        end
    end

    % Make the last row a rough RK weight vector summing to 1.
    w = rand(1,s);
    w = w/sum(w);
    R0(end,:) = w;
    R0(end,end) = max(R0(end,end), 0.05);

    % Occasionally perturb from a smaller embedded pattern.
    if mod(startId,5) == 0
        R0 = tril(0.2*randn(s));
        R0(1:s+1:end) = gamma;
        R0(end,:) = ones(1,s)/s;
    end

    z0 = packLowerTriangular(R0);
end

function [Atilde, Rtilde, S] = shiftedMatrices(R)
    s = size(R,1);
    S = full(spdiags(ones(s,1), -1, s, s));
    Atilde = S*R;
    Rtilde = R - Atilde;
end

function res = imexRK3ResidualCommonWeights(A, Atilde, b)
    s = size(A,1);
    e = ones(s,1);
    c = A*e;
    ctilde = Atilde*e;

    res = [
        b.'*e - 1

        b.'*c - 1/2
        b.'*ctilde - 1/2

        b.'*(c.^2) - 1/3
        b.'*(c.*ctilde) - 1/3
        b.'*(ctilde.^2) - 1/3

        b.'*A*c - 1/6
        b.'*A*ctilde - 1/6
        b.'*Atilde*c - 1/6
        b.'*Atilde*ctilde - 1/6
    ];
end

function z = packLowerTriangular(R)
    s = size(R,1);
    z = zeros(s*(s+1)/2,1);
    p = 0;
    for i = 1:s
        for j = 1:i
            p = p + 1;
            z(p) = R(i,j);
        end
    end
end

function R = unpackLowerTriangular(z, s)
    R = zeros(s);
    p = 0;
    for i = 1:s
        for j = 1:i
            p = p + 1;
            R(i,j) = z(p);
        end
    end
end

