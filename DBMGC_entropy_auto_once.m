function [y, w, objHistory] = DBMGC_entropy_auto_once(Ls, Y, e_type)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%

nKernel = length(Ls);
nCluster = size(Y, 2);
%**************************************************************************
% Initialization w and Y
%**************************************************************************
w = update_w(ones(nKernel, 1));
Lw = compute_Ls(Ls, w);
o1 = sum(sum(Y .* (Lw * Y)));
o2 = generalized_entropy(sum(Y)', e_type);
objHistory = [];
for iter = 1:50
    %**********************************************************************
    % Update lambda, fix Y, w;
    %**********************************************************************
    
    lambda = o2 / (2 * o1);

    %**********************************************************************
    % Update Y, fix w, lambda;
    %**********************************************************************
    [Y, obj_Y] = solve_Y_entropy_auto_once(Lw, Y, lambda, e_type);

    %**********************************************************************
    % Update w, Yix Y;
    %**********************************************************************
    e = compute_err(Ls, Y);
    w = update_w(e);
    Lw = compute_Ls(Ls, w);

    o1 = sum(sum(Y .* (Lw * Y)));
    o2 = generalized_entropy(sum(Y)', e_type);
    obj = lambda^2 * o1 - lambda * o2;
    objHistory = [objHistory; obj]; %#ok
    if iter > 2 && abs(objHistory(iter - 1) - objHistory(iter)) / abs(objHistory(iter - 1)) < 1e-10
        break;
    end
    
end
y = vec2ind(Y')';
end

function e = compute_err(Ls, Y)
nKernel = length(Ls);
e = zeros(nKernel, 1);
for iKernel = 1:nKernel
    LY = Ls{iKernel} * Y;
    e(iKernel) = sum(sum(Y .* LY));
end
end

function Lw = compute_Ls(Ls, w)
Lw = zeros(size(Ls{1}, 1));
for iKernel = 1:length(Ls)
    Lw = Lw + w(iKernel) * Ls{iKernel};
end
Lw = (Lw + Lw')/2;
end
