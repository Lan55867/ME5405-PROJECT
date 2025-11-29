% =========================================================================
% SOM + HOG 特征
% =========================================================================

clear; clc; close all;
rng(42); % 固定随机种子

datasetPath = "C:\Users\LENOVO\Desktop\5405\data\p_dataset_26";

if ~exist(datasetPath, 'dir')
    error('❌ 路径错误: 找不到文件夹 %s', datasetPath);
end

% HOG 参数
targetSize = [32, 32]; 
hogCellSize = [8, 8];  

%% ========== 1. 加载训练数据并提取 HOG ==========
fprintf('>>> [步骤 1] 加载训练数据并提取 HOG 特征...\n');
[X_train, Y_train_str] = helperLoadHOGData(datasetPath, targetSize, hogCellSize);

% 将标签转换为数字索引
categories = {'1', '2', '3', 'A', 'B', 'C'};
Y_train_idx = zeros(1, length(Y_train_str));
for i = 1:length(categories)
    Y_train_idx(Y_train_str == categories{i}) = i;
end

fprintf('  > 特征加载完成。维度: %d x %d\n', size(X_train, 1), size(X_train, 2));

%% ========== 2. 训练 SOM ==========
fprintf('>>> [步骤 2] 训练 SOM 网络 (HOG特征)... \n');

% SOM 参数
mapSize = [10, 10];    % 100 个神经元
nb_iter = 10000;        % 迭代次数
eff_width_init = 5;
eff_width_time_cst = nb_iter / log(eff_width_init);
lr_init = 0.5;
lr_time_cst = nb_iter;

% 训练
weights = trainSOM_Core(X_train, mapSize(1), mapSize(2), nb_iter, ...
    eff_width_init, eff_width_time_cst, lr_init, lr_time_cst);

%% ========== 3. 标记神经元 ==========
fprintf('>>> [步骤 3] 标记神经元...\n');

nb_neurons = prod(mapSize);
neuron_labels = zeros(nb_neurons, 1);

for j = 1:nb_neurons
    % 找到离当前神经元权重最近的训练样本
    [~, winner_idx] = find_winner(X_train, weights(:, j));
    neuron_labels(j) = Y_train_idx(winner_idx);
end

%% ========== 4. 测试 Image 2 并可视化 ==========
fprintf('>>> [步骤 4] 测试 Image 2...\n');

try
    img_raw = readEncodedImage('charact1.txt');
catch
    error('❌ 找不到 charact1.txt');
end

% 分割与排序
stats = helperSegmentAndSort(img_raw);

% 准备结果显示
figure('Name', 'SOM+HOG Verification', 'Color', 'w');
final_str = "";

for i = 1:6
    % 1. 裁剪与预处理
    bbox = stats(i).BoundingBox;
    char_crop = imcrop(double(img_raw)/31.0, bbox);
    
    % 2. 提取 HOG
    [hog_vec, img_display] = getHOG_v5(char_crop, targetSize, hogCellSize);
    
    % 3. SOM 预测
    % find_winner 期望列向量，hog_vec 是行向量，需要转置
    [~, winner_idx] = find_winner(weights, hog_vec'); 
    pred_idx = neuron_labels(winner_idx);
    pred_char = categories{pred_idx};
    
    final_str = final_str + pred_char;
    
    % 4. 绘图
    subplot(2, 3, i);
    imshow(img_display);
    title(sprintf('预测: %s', pred_char), 'FontSize', 14, 'Color', 'blue', 'FontWeight', 'bold');
end

sgtitle({'SOM + HOG 验证结果'; ['识别序列: ' char(final_str)]});

fprintf('\n=======================================\n');
fprintf('🔮 最终识别结果: %s\n', final_str);
fprintf('   (如果是 123ABC，则验证成功！)\n'); 
fprintf('=======================================\n');


%% ============================================================
%                   辅助函数库
% ============================================================

function weights = trainSOM_Core(input_data, N, M, nb_iter, sig0, tau1, lr0, tau2)
    % SOM 核心训练逻辑
    [p, nb_samples] = size(input_data);
    weights = rand(p, N*M);
    
    h_wait = waitbar(0, 'SOM 训练中...');
    for t = 1:nb_iter
        % 随机采样
        x = input_data(:, randi(nb_samples));
        
        % 寻找 BMU (欧氏距离)
        dists = sum((weights - x).^2, 1);
        [~, winner_idx] = min(dists);
        
        % 更新参数
        lr = lr0 * exp(-(t-1)/tau2);
        sig = sig0 * exp(-(t-1)/tau1);
        
        % 计算邻域
        [win_i, win_j] = ind2sub([N, M], winner_idx);
        [grid_i, grid_j] = ind2sub([N, M], 1:N*M);
        dist_sq = (grid_i - win_i).^2 + (grid_j - win_j).^2;
        h_func = exp(-dist_sq / (2 * sig^2));
        
        % 更新权重
        weights = weights + lr .* h_func .* (x - weights);
        
        if mod(t, 500)==0, waitbar(t/nb_iter, h_wait); end
    end
    close(h_wait);
end

function [val, idx] = find_winner(weights, x)
    [val, idx] = min(sum((weights - x).^2, 1));
end

function [X, Y] = helperLoadHOGData(dataDir, targetSize, hogCellSize)
    classes = {'1', '2', '3', 'A', 'B', 'C'};
    X_list = {}; Y_list = {};
    for k = 1:6
        folder = fullfile(dataDir, classes{k});
        files = dir(fullfile(folder, '*.mat'));
        for i = 1:length(files)
            try
                d = load(fullfile(folder, files(i).name));
                fn = fieldnames(d);
                img = im2double(d.(fn{1}));
                [hog, ~] = getHOG_v5(img, targetSize, hogCellSize);
                X_list{end+1} = hog; % 行向量
                Y_list{end+1} = classes{k};
            catch; end
        end
    end
    X = vertcat(X_list{:})'; % 转置为: 特征 x 样本 (SOM 格式)
    Y = string(Y_list);
end

function [hog, img_out] = getHOG_v5(img, targetSize, hogCellSize)
    % 预处理：保持宽高比 + HOG
    if ndims(img) == 3, img = rgb2gray(img); end
    [h, w] = size(img);
    padSize = max(h, w);
    padded = zeros(padSize);
    r = floor((padSize-h)/2)+1; c = floor((padSize-w)/2)+1;
    padded(r:r+h-1, c:c+w-1) = img;
    img_out = imresize(padded, targetSize);
    [hog, ~] = extractHOGFeatures(img_out, 'CellSize', hogCellSize);
end

function img = readEncodedImage(filename)
    fid = fopen(filename, 'r'); raw = fscanf(fid, '%c'); fclose(fid);
    clean = raw(ismember(raw, ['0':'9', 'A':'V']));
    A = reshape(clean(1:4096), [64, 64])';
    img = zeros(64, 64);
    l = (A>='A'&A<='V'); img(l)=double(A(l))-55;
    d = (A>='0'&A<='9'); img(d)=double(A(d))-48;
    img = uint8(img);
end

function stats_sorted = helperSegmentAndSort(img)
    bw = img > 0;
    stats = regionprops(bwconncomp(bw), 'Area', 'Centroid', 'BoundingBox');
    [~, idx] = sort([stats.Area], 'descend'); stats = stats(idx(1:6));
    
    cen = vertcat(stats.Centroid);
    [~, y_idx] = sort(cen(:,2));
    top = stats(y_idx(1:3));    % 上行 1,2,3
    bot = stats(y_idx(4:6));    % 下行 A,B,C
    
    c_t = vertcat(top.Centroid); [~, xt] = sort(c_t(:,1));
    c_b = vertcat(bot.Centroid); [~, xb] = sort(c_b(:,1));
    
    stats_sorted = [top(xt); bot(xb)];
end