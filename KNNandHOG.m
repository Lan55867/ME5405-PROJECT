% =========================================================================
% kNN + HOG
% =========================================================================

clear; clc; close all;
rng(42); % 固定随机种子

%% ========== 1. 配置与数据加载 ==========
fprintf('>>> [步骤 1] 正在加载训练数据 (p_dataset_26)...\n');

datasetPath = "C:\Users\LENOVO\Desktop\5405\data\p_dataset_26";

if ~exist(datasetPath, 'dir')
    error('❌ 路径错误: 找不到文件夹 %s', datasetPath);
end

% 设定参数
targetSize = [32, 32];  
hogCellSize = [8, 8];   

% 加载并提取特征
[X_full, Y_full] = helperLoadTrainingData(datasetPath, targetSize, hogCellSize);

% 分割训练/验证集 (75% / 25%)
cv = cvpartition(Y_full, 'HoldOut', 0.25);
X_train = X_full(cv.training, :);
Y_train = Y_full(cv.training);
X_val   = X_full(cv.test, :);
Y_val   = Y_full(cv.test);

fprintf('  > 训练集: %d, 验证集: %d\n', length(Y_train), length(Y_val));

%% ========== 2. 训练与调优 kNN 模型 (Task 9) ==========
fprintf('>>> [步骤 2] 正在寻找最佳 k 值...\n');

k_values = [1, 3, 5, 7, 9]; % 尝试不同的邻居数量
best_acc = 0;
best_k = 1;
best_model = [];

for k = k_values
    % 训练 kNN 模型
    % Distance: 'euclidean' (欧氏距离) 
    % Standardize: false (HOG 已经归一化了，不需要再次标准化)
    knn_model = fitcknn(X_train, Y_train, ...
        'NumNeighbors', k, ...
        'Distance', 'euclidean', ...
        'Standardize', false);
    
    % 验证
    YPred_val = predict(knn_model, X_val);
    acc = mean(string(YPred_val) == string(Y_val));
    
    fprintf('  k=%d, 验证准确率: %.2f%%\n', k, acc * 100);
    
    if acc >= best_acc
        best_acc = acc;
        best_k = k;
        best_model = knn_model;
    end
end

fprintf('✅ 最佳模型选定: k=%d (准确率 %.2f%%)\n', best_k, best_acc * 100);

%% ========== 3. 处理测试图像 (Image 2) ==========
fprintf('>>> [步骤 3] 处理测试图像 (charact1.txt)...\n');

try
    img_raw = readEncodedImage('charact1.txt');
catch
    error('❌ 找不到 charact1.txt，请确保文件在当前目录下');
end

% 分割与排序
stats = helperSegmentAndSort(img_raw);

% 提取测试集 HOG 特征
X_test = zeros(6, size(X_train, 2),'single');
test_imgs_display = cell(1, 6); 

for i = 1:6
    % 裁剪
    bbox = stats(i).BoundingBox;
    % 归一化并裁剪
    char_crop = imcrop(double(img_raw)/31.0, bbox);
    
    % 预处理 (Padding + Resize) & HOG
    [hog_feat, img_processed] = getHOG_v5(char_crop, targetSize, hogCellSize);
    
    X_test(i, :) = hog_feat;
    test_imgs_display{i} = img_processed;
end

%% ========== 4. 最终预测与可视化 ==========
fprintf('>>> [步骤 4] 最终预测...\n');

YPred_test = predict(best_model, X_test);
result_str = strjoin(string(YPred_test), '');

fprintf('\n=======================================\n');
fprintf('🔮 kNN 最终识别结果: %s\n', result_str);
fprintf('   (正确结果应为: 123ABC)\n'); 
fprintf('=======================================\n');

% 可视化
figure('Name', 'Final Prediction (kNN)', 'Color', 'w');
for i = 1:6
    subplot(2, 3, i);
    imshow(test_imgs_display{i});
    title(sprintf('预测: %s', string(YPred_test(i))), 'FontSize', 14, 'Color', 'b');
end
sgtitle(sprintf('kNN (k=%d) 识别结果', best_k));


%% ============================================================
%                   辅助函数库
% ============================================================

% 1. 数据加载与 HOG 提取
function [X, Y] = helperLoadTrainingData(dataDir, targetSize, hogCellSize)
    classes = {'1', '2', '3', 'A', 'B', 'C'};
    X_list = {}; Y_list = {};
    h = waitbar(0, '正在加载 .mat 数据...');
    for k = 1:length(classes)
        label = classes{k};
        folder = fullfile(dataDir, label);
        files = dir(fullfile(folder, '*.mat'));
        for i = 1:length(files)
            try
                d = load(fullfile(folder, files(i).name));
                fn = fieldnames(d);
                img = im2double(d.(fn{1}));
                [hog, ~] = getHOG_v5(img, targetSize, hogCellSize);
                X_list{end+1} = hog;
                Y_list{end+1} = label;
            catch; continue; end
        end
        waitbar(k/6, h);
    end
    close(h);
    X = vertcat(X_list{:});
    Y = string(Y_list');
end

% 2. 图像读取
function img = readEncodedImage(filename)
    fid = fopen(filename, 'r');
    if fid == -1, error('无法打开文件'); end
    raw = fscanf(fid, '%c'); fclose(fid);
    clean = raw(ismember(raw, ['0':'9', 'A':'V']));
    A = reshape(clean(1:4096), [64, 64])';
    img = zeros(64, 64);
    l = (A>='A'&A<='V'); img(l)=double(A(l))-55;
    d = (A>='0'&A<='9'); img(d)=double(A(d))-48;
    img = uint8(img);
end

% 3. 分割与排序
function stats_sorted = helperSegmentAndSort(img_uint8)
    bw = img_uint8 > 0;
    cc = bwconncomp(bw);
    stats = regionprops(cc, 'Area', 'Centroid', 'BoundingBox');
    [~, idx] = sort([stats.Area], 'descend'); stats = stats(idx(1:6));
    
    cen = vertcat(stats.Centroid);
    [~, y_idx] = sort(cen(:, 2));
    row_top = stats(y_idx(1:3));    % 上行 123
    row_bot = stats(y_idx(4:6));    % 下行 ABC
    
    c_t = vertcat(row_top.Centroid); [~, xt] = sort(c_t(:, 1));
    c_b = vertcat(row_bot.Centroid); [~, xb] = sort(c_b(:, 1));
    
    stats_sorted = [row_top(xt); row_bot(xb)]; 
end

% 4. HOG 特征提取
function [hog, img_out] = getHOG_v5(img, targetSize, hogCellSize)
    if ndims(img) == 3, img = rgb2gray(img); end
    [h, w] = size(img);
    padSize = max(h, w);
    padded = zeros(padSize);
    r = floor((padSize-h)/2)+1; c = floor((padSize-w)/2)+1;
    padded(r:r+h-1, c:c+w-1) = img;
    img_out = imresize(padded, targetSize);
    [hog, ~] = extractHOGFeatures(img_out, 'CellSize', hogCellSize);
end