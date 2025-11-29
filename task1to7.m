% =========================================================================
% ME5405 机器视觉 - 图像 2 (Characters) - 任务 1-7 
%
% 目标:
% 1. 显示原始图像 [cite: 438]
% 2. 创建二值图像 [cite: 439]
% 3. 创建单像素细化图像 [cite: 440]
% 4. 确定轮廓 [cite: 441]
% 5. 分割和标记字符 [cite: 442]
% 6. 按 "AB123C" 顺序重排 [cite: 443]
% 7. 将重排后的图像旋转 30 度 [cite: 444]
%
% 需要: 
% 1. Image Processing Toolbox
% 2. 文件 'readAlphanumericImage.m' (在同一文件夹中)
% 3. 文件 'charact1.txt' (在同一文件夹中)
% =========================================================================

clc;        % 清空命令行
clear;      % 清空工作区变量
close all;  % 关闭所有图像窗口

%% --- 任务 1: 显示原始图像 --- [cite: 438]
fprintf('>>> 正在执行任务 1: 显示原始图像...\n');

% 使用辅助函数读取 64x64 的 .txt 图像数据
% 该图像是 32 级灰度 (0-31) [cite: 366]
try
    img_original = help_task1_7('charact1.txt');
catch ME
    error('❌ 无法读取 charact1.txt: %s\n请确保它与此 .m 文件在同一文件夹中。', ME.message);
end

figure;
% '[]' 自动将 0-31 的灰度范围缩放到 0-255 进行显示
imshow(img_original, []); 
title('任务 1: 原始图像 (Image 2)');


%% --- 任务 2: 创建二值图像 --- [cite: 439]
fprintf('>>> 正在执行任务 2: 创建二值图像...\n');

% [关键] 对于 0-31 编码的图像 [cite: 366]，0 是背景，任何 > 0 都是字符。
% 自动阈值 imbinarize(img_original) 在此会失败。
img_bw = img_original > 0;

figure;
imshow(img_bw);
title('任务 2: 二值图像 (Thresholded)');


%% --- 任务 3: 确定单像素细化图像 --- [cite: 440]
fprintf('>>> 正在执行任务 3: 创建单像素细化图像...\n');

% bwmorph 的 'thin' 操作执行骨架化
% 'Inf' 表示无限次迭代，直到收敛
img_thin = bwmorph(img_bw, 'thin', Inf);

figure;
imshow(img_thin);
title('任务 3: 单像素细化图像 (Skeleton)');



%% --- 任务 4: 确定轮廓 --- [cite: 441]
fprintf('>>> 正在执行任务 4: 确定轮廓...\n');

% bwperim 查找二值图像中对象的轮廓
% 注意：这是在原始二值图 (img_bw) 上操作，而不是细化图上
img_outline = bwperim(img_bw);

figure;
imshow(img_outline);
title('任务 4: 图像轮廓 (Outlines)');


fprintf('>>> 正在执行任务 6: 重排字符...\n');

% 1. 提取每个字符的属性
cc = bwconncomp(img_bw);
stats = regionprops(cc, 'Image', 'Centroid', 'Area', 'BoundingBox');

% 2. 过滤掉噪声 (按面积取最大的 6 个)
if isempty(stats)
    error('Image 2 分割失败：未找到任何对象。');
end
[~, areaIdx] = sort([stats.Area], 'descend');
numToKeep = min(6, length(stats));
if numToKeep < 6
    error('Image 2 分割失败：只找到 %d 个对象（预期 6 个）。', numToKeep);
end
stats = stats(areaIdx(1:numToKeep)); % 只保留 6 个最大的

% 3. 稳健排序逻辑 (先按 Y 排序分行，再按 X 排序分列)
centroids = vertcat(stats.Centroid);

% 3.1 按 Y 坐标排序 (区分上下行)
% Y 轴向下为正。
% 根据你的错误输出，数字(123)在上方(Y小)，字母(ABC)在下方(Y大)
[~, sort_y_idx] = sort(centroids(:, 2));
stats_y_sorted = stats(sort_y_idx);

% 前 3 个是上行 (根据你的图像证据，这是 1, 2, 3)
row_top = stats_y_sorted(1:3);    

% 后 3 个是下行 (根据你的图像证据，这是 A, B, C)
row_bottom = stats_y_sorted(4:6); 

% 3.2 分别对每行按 X 坐标排序 (从左到右)
centroids_top = vertcat(row_top.Centroid);
[~, sort_x_top_idx] = sort(centroids_top(:, 1));
row_top_sorted = row_top(sort_x_top_idx); % 顺序: 1, 2, 3

centroids_bottom = vertcat(row_bottom.Centroid);
[~, sort_x_bottom_idx] = sort(centroids_bottom(:, 1));
row_bottom_sorted = row_bottom(sort_x_bottom_idx); % 顺序: A, B, C

% 4. 提取图像并按 "AB123C" 组合
% [修正]: 交换了分配逻辑
char_1 = row_top_sorted(1).Image;
char_2 = row_top_sorted(2).Image;
char_3 = row_top_sorted(3).Image;

char_A = row_bottom_sorted(1).Image;
char_B = row_bottom_sorted(2).Image;
char_C = row_bottom_sorted(3).Image;

chars_to_arrange = {char_A, char_B, char_1, char_2, char_3, char_C};

padding = 10; % 在字符之间留 10 像素的空白

% 计算新画布的尺寸
max_h = 0;
total_w = padding; % 从左侧的 padding 开始
for i = 1:length(chars_to_arrange)
    max_h = max(max_h, size(chars_to_arrange{i}, 1));
    total_w = total_w + size(chars_to_arrange{i}, 2) + padding;
end
max_h = max_h + 2*padding; % 上下也添加 padding

% 创建新画布
img_arranged = false(max_h, total_w);

% 将每个字符"粘贴"到画布上
current_x = padding + 1;
for i = 1:length(chars_to_arrange)
    img = chars_to_arrange{i};
    [h, w] = size(img);
    
    % 计算 Y 起始位置，使字符垂直居中
    y_start = floor((max_h - h) / 2) + 1;
    
    % 放置图像
    img_arranged(y_start : y_start+h-1, current_x : current_x+w-1) = img;
    
    % 更新下一个字符的 X 起始位置
    current_x = current_x + w + padding;
end

figure;
imshow(img_arranged);
title('任务 6: 重排为 "AB123C"');


%% --- 任务 7: 旋转图像 30 度 ---
fprintf('>>> 正在执行任务 7: 旋转图像 30 度...\n');

% 项目要求 "旋转 30 度"。正值在 imrotate 中表示逆时针。
% [修正]: 将 'crop' 改为 'loose'，以防止图像被裁剪
img_rotated = imrotate(img_arranged, 30, 'bicubic', 'loose');

figure;
imshow(img_rotated);
title('任务 7: 旋转 30 度 (完整视图)');

fprintf('>>> 图像 2 的任务 1-7 已全部完成。\n');

