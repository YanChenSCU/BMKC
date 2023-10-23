%
%
%
clear;
clc;
data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = fullfile(pwd, '..',  filesep, "BMKC", filesep);
addpath(code_path);


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

datasetCandi = {'FACS_v2_Trachea-counts_1013n_13741d_7c_uni.mat','hitech_2301n_22498d_6c_tfidf_uni.mat',...
    'k1b_2340n_21839d_6c_tfidf_uni.mat','FACS_v2_Large_Intestine-counts_3362n_16418d_15c_uni.mat',...
    'FACS_v2_Fat_3618n_15492d_9c_uni.mat',   'MNIST_4000n_784d_10c_uni.mat',...
    'Macosko_6418n_8608d_39c_uni.mat','caltech101_silhouettes_8671n_784d_101c_28_uni.mat'};
exp_n = 'agtBMKC12';
% profile off;
% profile on;
for i1 = 1%1 : length(datasetCandi)%
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    try
        if ~exist(dir_name, 'dir')
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end

    clear X y Y;
    load(data_name);
    if exist('y', 'var')
        Y = y;
    end
    if size(X, 1) ~= size(Y, 1)
        Y = Y';
    end
    assert(size(X, 1) == size(Y, 1));
    nSmp = size(X, 1);
    nCluster = length(unique(Y));

    %*********************************************************************
    % BMKC
    %*********************************************************************
    fname2 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '.mat']);
    if ~exist(fname2, 'file')
        Xs = cell(1,1);
        Xs{1} = X;
        Ks = Xs_to_Ks_12k(Xs);
        Ks2 = Ks{1,1};
        Ks = Ks2;
        clear Ks2 Xs;
        [nSmp, ~, nKernel] = size(Ks);

        %**************************************************************************
        % Parameter Configuration
        %**************************************************************************
        nRepeat = 10;
        k_range = [5,10,15,20];
        t_range = [1,3,5,7,9];
        entropy_range = 12;
        nMeasure = 13;

        %**************************************************************************
        % Construct As from Ks
        %**************************************************************************
        iParam = 0;
        nParam = length(k_range) * length(t_range) * length(entropy_range);
        agtBMKC12_result = zeros(nParam, 1, nRepeat, nMeasure);
        agtBMKC12_time = zeros(nParam, 1);


        for iKnn = 1:length(k_range)
            tic;
            knn_size = k_range(iKnn);
            Ls_0 = Ks2Ls2(Ks, knn_size);
            t0 = toc;
            for it = 1:length(t_range)
                tic;
                t = t_range(it);
                Ls = cell(nKernel, 1);
                for iKernel = 1:nKernel
                    S = expm(- t * Ls_0{iKernel});
                    S = S - 1e8 * eye(nSmp);
                    [Val, Idx] = sort(S, 2, 'descend');
                    Idx = Idx(:, 1:knn_size);
                    Val = Val(:, 1:knn_size);
                    SS = zeros(nSmp);
                    for iSmp = 1:nSmp
                        idxa0 = Idx(iSmp, :);
                        ad = Val(iSmp, :);
                        SS(iSmp, idxa0) = EProjSimplex_new(ad);
                    end
                    Li = diag(sum(SS, 1)) + diag(sum(SS, 2)) - SS - SS';
                    Li = (Li + Li')/2;
                    Ls{iKernel} = Li;
                end
                %**************************************************************************
                % Initialization w
                %**************************************************************************
                w0 = update_w(ones(nKernel, 1));
                Lw = zeros(nSmp);
                for iKernel = 1:nKernel
                    Lw = Lw + w0(iKernel) * Ls{iKernel};
                end
                Lw = (Lw + Lw')/2;

                %**************************************************************************
                % Initialization Y0
                %**************************************************************************
                opt.disp = 0;
                [H, ~] = eigs(Lw, nCluster,'SA',opt);
                H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,nCluster);
                t1 = toc;

                for iEntropy = 1:length(entropy_range)
                    iParam = iParam + 1;
                    disp(['BMKC iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
                    fname3 = fullfile(prefix_mdcs, [data_name, '_12k_', exp_n, '_', num2str(iParam), '.mat']);
                    if exist(fname3, 'file')
                        load(fname3, 'result_11_s', 't0', 't1', 't2');
                        agtBMKC12_time(iParam) = t0 + t1 + t2/nRepeat;
                        for iRepeat = 1:nRepeat
                            agtBMKC12_result(iParam, 1, iRepeat, :) = result_11_s(iRepeat, :);
                        end
                    else
                        result_11_s = zeros(nRepeat, nMeasure);
                        tic;
                        for iRepeat = 1:nRepeat
                            % label0 = litekmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 1);
                            label0 = kmeans(H_normalized, nCluster, 'MaxIter', 50, 'Replicates', 10);
                            Y0 = full(ind2vec(label0'))';
                            e_type = entropy_range(iEntropy);
                            %**************************************************************************
                            % Update w
                            %**************************************************************************
                            [label, w, obj] = DBMGC_entropy_auto_once(Ls, Y0, e_type);

                            result_11 = my_eval_y(label, Y);
                            result_11_s(iRepeat, :) = result_11';
                            agtBMKC12_result(iParam, 1, iRepeat, :) = result_11';
                        end
                        t2 = toc;
                        agtBMKC12_time(iParam) = t0 + t1 + t2/nRepeat;
                        save(fname3, 'result_11_s', 't0', 't2', 't1', 'knn_size', 't', 'e_type');
                    end
                end
            end
        end
        a1 = sum(agtBMKC12_result, 2);
        a3 = sum(a1, 3);
        a4 = reshape(a3, size(agtBMKC12_result,1), size(agtBMKC12_result,4));
        agtBMKC12_grid_result = a4/nRepeat;
        agtBMKC12_result_summary = [max(agtBMKC12_grid_result, [], 1), sum(agtBMKC12_time)/nParam];
        save(fname2, 'agtBMKC12_result', 'agtBMKC12_grid_result', 'agtBMKC12_time', 'agtBMKC12_result_summary');
        disp([data_name, ' has been completed!']);
    end
end

rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);

% profile viewer