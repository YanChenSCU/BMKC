function Ls = Ks2Ls2(Ks, k)
[nSmp, ~, nKernel] = size(Ks);
avgK = sum(Ks, 3) - 10^8*eye(nSmp);
[~, Idx] = sort(avgK, 1, 'descend');
Idx = Idx(1:k, :);
Idx2 = Idx';
colIdx = Idx2(:);

e = ones(1, k);
z = zeros(k, 1);
e2 = ones(k, 1);
options = [];
options.Display = 'off';
Idxs = zeros(k, nSmp, nKernel);
for i1 = 1:nKernel
    Idxs(:, :, i1) = Idx;
end
Ik = eye(k);
As = zeros(nSmp, k, nKernel);
Ls = cell(1, nKernel);
for i1 = 1:nKernel
    %**********************************************
    %  Step1:SK-LKR
    %  Complexity
    %         (1)avgKernel, n * n addition
    %         (2)knn, m * n * n, top-k quick selection is O(n)
    %         (3)quadprog, m * n k3
    %
    %**********************************************
    Ai = As(:, :, i1);
    Ki = Ks(:, :, i1);
    Idxi = Idxs(:, :, i1);
    for iSmp = 1:nSmp
        idx = Idxi(:, iSmp); 
        ki = Ki(idx, iSmp);
        Kii = Ki(idx, idx') + Ik;
        v = quadprog(Kii, -ki, [], [], e, 1, z, e2, [], options);
        Ai(iSmp, :) = v;
    end
    rowIdx = repmat((1:nSmp)', k, 1);
    val = Ai(:);
    G = sparse(rowIdx, colIdx, val, nSmp, nSmp,nSmp * k);

    L = diag(sum(G,1)) + diag(sum(G,2)) - G - G';
    Ls{i1} = L;
end
