function [y, objHistory] = DBSGC_entropy_auto_once(L, Y, e_type)
% DBMGC  Discrete Balanced Multiple Graph Clustering.
%   [y, w, obj] = DBMGC(K, Y)
%   K: n*n kernel matrix.
%   Y: n*c initial label indicator matrix.
%


nCluster = size(Y, 2);
%**************************************************************************
% Initialization w and Y
%**************************************************************************
o1 = sum(sum(Y .* (L * Y)));
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
    [Y, obj_Y] = solve_Y_entropy_auto_once(L, Y, lambda, e_type);



    o1 = sum(sum(Y .* (L * Y)));
    o2 = generalized_entropy(sum(Y)', e_type);
    obj = lambda^2 * o1 - lambda * o2;
    objHistory = [objHistory; obj]; %#ok
    if iter > 2 && abs(objHistory(iter - 1) - objHistory(iter)) / abs(objHistory(iter - 1)) < 1e-10
        break;
    end
    
end
y = vec2ind(Y')';
end