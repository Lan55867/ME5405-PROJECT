% =========================================================================
% ME5405 项目 – 最终完整解决方案 (SVM + HOG)
% 使用 Linear 核
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
targetSize = [32, 32];  % 统一尺寸
hogCellSize = [8, 8];   % 提取粗粒度特征，泛化性更好
kernelType = 'linear';  % 使用Linear，鲁棒性更强

% 加载并提取训练特征
[X_full, Y_full] = helperLoadTrainingData(datasetPath, targetSize, hogCellSize);

% 分割训练/验证集 (75% / 25%)
cv = cvpartition(Y_full, 'HoldOut', 0.25);
X_train = X_full(cv.training, :);
Y_train = Y_full(cv.training);
X_val   = X_full(cv.test, :);
Y_val   = Y_full(cv.test);

fprintf('  > 训练集: %d, 验证集: %d\n', length(Y_train), length(Y_val));

%% ========== 2. 训练 SVM 模型 ==========
fprintf('>>> [步骤 2] 正在训练 SVM (%s 核)...\n', kernelType);

% 关键设定：
% 1. Standardize = false (HOG 已经归一化了，双重归一化会破坏特征)
% 2. Kernel = linear (简单模型通常对字符识别更有效)
t = templateSVM('KernelFunction', kernelType, 'Standardize', false);

% 训练多分类模型
SVMModel = fitcecoc(X_train, Y_train, 'Learners', t);

% 验证集准确率
YPred_val = predict(SVMModel, X_val);
valAcc = mean(string(YPred_val) == string(Y_val));
fprintf('  ✅ 验证集准确率: %.2f%%\n', valAcc * 100);

%% ========== 3. 处理测试图像 (Image 2) ==========
fprintf('>>> [步骤 3] 处理测试图像 (charact1.txt)...\n');

try
    img_raw = readEncodedImage('charact1.txt');
catch
    error('❌ 找不到 charact1.txt，请确保文件在当前目录下');
end

% 归一化 (0-31 -> 0-1)
img_normalized = double(img_raw) / 31.0;

% 分割与排序 (包含 Task 6 的修正逻辑)
stats = helperSegmentAndSort(img_raw);

% 提取测试集 HOG 特征 (使用 v5 填充逻辑)
X_test = zeros(6, size(X_train, 2));
test_imgs_display = cell(1, 6); % 用于展示

for i = 1:6
    % 裁剪
    bbox = stats(i).BoundingBox;
    char_crop = imcrop(img_normalized, bbox);
    
    % 预处理 (Padding + Resize) & HOG
    [hog_feat, img_processed] = getHOG_v5(char_crop, targetSize, hogCellSize);
    
    X_test(i, :) = hog_feat;
    test_imgs_display{i} = img_processed;
end

%% ========== 4. 最终预测与可视化 ==========
fprintf('>>> [步骤 4] 最终预测...\n');

YPred_test = predict(SVMModel, X_test);
result_str = strjoin(string(YPred_test), '');

fprintf('\n=======================================\n');
fprintf('🔮 最终识别结果: %s\n', result_str);
fprintf('   (正确结果应为: 123ABC)\n'); 
fprintf('=======================================\n');

% 可视化
figure('Name', 'Final Prediction', 'Color', 'w');
for i = 1:6
    subplot(2, 3, i);
    imshow(test_imgs_display{i});
    title(sprintf('预测: %s', string(YPred_test(i))), 'FontSize', 14, 'Color', 'b');
end
sgtitle(sprintf('SVM (%s) 识别结果', kernelType));


%% ============================================================
%                   Helper Functions
% ============================================================

% 1. 数据加载与 HOG 提取
function [X, Y] = helperLoadTrainingData(dataDir, targetSize, hogCellSize)
    classes = {'1', '2', '3', 'A', 'B', 'C'};
    X_list = {}; Y_list = {};
    
    % 进度条
    h = waitbar(0, '正在加载 .mat 数据...');
    
    for k = 1:length(classes)
        label = classes{k};
        folder = fullfile(dataDir, label);
        files = dir(fullfile(folder, '*.mat'));
        
        for i = 1:length(files)
            try
                d = load(fullfile(folder, files(i).name));
                fn = fieldnames(d);
                img = d.(fn{1}); % 读取图像
                
                % 归一化并转为 double (0-255 -> 0-1)
                img = im2double(img);
                
                % 提取 HOG (使用 v5 填充逻辑)
                [hog, ~] = getHOG_v5(img, targetSize, hogCellSize);
                
                X_list{end+1} = hog;
                Y_list{end+1} = label;
            catch
                continue;
            end
        end
        waitbar(k/6, h);
    end
    close(h);
    X = vertcat(X_list{:});
    Y = string(Y_list');
end

% 2. 图像读取
function img = readEncodedImage(filename)
    % 鲁棒读取函数：自动跳过换行符和空格
    
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    % 1. 读取整个文件内容为一长串字符
    raw_text = fscanf(fid, '%c'); 
    fclose(fid);
    
    % 2. 删除所有非数据字符（换行符、回车符、空格）
    %    只保留 0-9 和 A-V
    %    ASCII 48-57 是 '0'-'9', 65-86 是 'A'-'V'
    clean_data = raw_text(ismember(raw_text, ['0':'9', 'A':'V']));
    
    % 3. 检查数据量是否足够
    expected_pixels = 64 * 64;
    if length(clean_data) < expected_pixels
        error('文件 %s 数据不足！只有 %d 个有效字符，需要 %d 个。', ...
            filename, length(clean_data), expected_pixels);
    end
    
    % 4. 截取前 4096 个字符并重塑矩阵
    %    MATLAB 是按列填充的，文本文件是按行写的
    %    所以需要先转置
    A_vec = clean_data(1:expected_pixels);
    A_matrix = reshape(A_vec, [64, 64])'; 
    
    % 5. 解码逻辑 (0-9 -> 0-9, A-V -> 10-31)
    img_out = zeros(64, 64);
    
    % 处理字母 A-V
    mask_letter = (A_matrix >= 'A' & A_matrix <= 'V');
    img_out(mask_letter) = double(A_matrix(mask_letter)) - double('A') + 10;
    
    % 处理数字 0-9
    mask_digit = (A_matrix >= '0' & A_matrix <= '9');
    img_out(mask_digit) = double(A_matrix(mask_digit)) - double('0');
    
    % 转换为 uint8
    img = uint8(img_out);
end

% 3. 分割与排序 (Task 6 修正版)
function stats_sorted = helperSegmentAndSort(img_uint8)
    bw = img_uint8 > 0; % 简单的阈值分割
    cc = bwconncomp(bw);
    stats = regionprops(cc, 'Area', 'Centroid', 'BoundingBox');
    
    % 过滤噪点 (取最大的 6 个)
    [~, idx] = sort([stats.Area], 'descend');
    stats = stats(idx(1:6));
    
    % --- 排序逻辑 ---
    centroids = vertcat(stats.Centroid);
    [~, y_idx] = sort(centroids(:, 2)); % 按 Y 排序
    
    row_top = stats(y_idx(1:3));    % 上行 (Y 小) -> 数字 123
    row_bottom = stats(y_idx(4:6)); % 下行 (Y 大) -> 字母 ABC
    
    % 行内按 X 排序
    cent_top = vertcat(row_top.Centroid);
    [~, x_top] = sort(cent_top(:, 1));
    row_top = row_top(x_top);
    
    cent_bot = vertcat(row_bottom.Centroid);
    [~, x_bot] = sort(cent_bot(:, 1));
    row_bottom = row_bottom(x_bot);
    
    % 最终顺序: 1, 2, 3, A, B, C
    
    stats_sorted = [row_top; row_bottom]; 
end

% 4. HOG 特征提取
function [hog, img_out] = getHOG_v5(img, targetSize, hogCellSize)
    % 1. 确保灰度
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    
    % 2. Padding (保持宽高比)
    % 这是解决拉伸变形问题的关键
    [h, w] = size(img);
    padSize = max(h, w);
    padded = zeros(padSize); % 黑色背景填充
    
    r_start = floor((padSize - h)/2) + 1;
    c_start = floor((padSize - w)/2) + 1;
    padded(r_start:r_start+h-1, c_start:c_start+w-1) = img;
    
    % 3. Resize 
    img_out = imresize(padded, targetSize);
    
    % 4. Extract HOG
    [hog, ~] = extractHOGFeatures(img_out, 'CellSize', hogCellSize);
end