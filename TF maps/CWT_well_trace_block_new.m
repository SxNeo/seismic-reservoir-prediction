clear all;
close all;
clc;

%%%%%%%%%%%%% 加载地震数据 %%%%%%%%%%%%%
[seismic, sampint, ch, bh, th] = altreadsegy('copy_post_stack_300_256_delta30.sgy', ...
    'textHeader', 'yes', 'textformat', 'ascii', 'fpformat', 'ibm', ...
    'binaryheader', 'yes', 'traceheaders', 'yes');

[nsample, ntrace] = size(seismic);

dt = 0.004; % 采样率（秒）
fs = 1 / dt; % 采样频率
t = 0:1:nsample-1;

%%%%%%%%%%%%% CWT 参数 %%%%%%%%%%%%%
wavename = 'cmor0.2-2';
Fc = centfrq(wavename); % 小波的中心频率
c = 2 * Fc * nsample;
scals = c ./ (1:nsample);
f = scal2frq(scals, wavename, 1 / fs); % 将尺度转换为频率

f_max = 80;
f_min = 10;
interval = 5;

% 找到最接近间隔为5Hz的频率对应的索引
f_sampled = f_min:interval:f_max;
idx_sampled = arrayfun(@(freq) find(abs(f - freq) == min(abs(f - freq)), 1), f_sampled);

well_trace = [63 64 65 240 241 242];
for i = well_trace

    waitbar(i / ntrace);

    x = seismic(:, i);
    x = x / max(abs(x));

    coefs = cwt(x, scals, wavename); % 计算连续小波系数

    %% 设置时频谱范围
    max_value = max(max(abs(coefs)));
    min_value = min(min(abs(coefs)));

    %% 单道时频谱
%     figure
%     img = abs(coefs(idx_sampled, :))';
%     img_resized = imresize(img, [size(img, 1), 63]); % 将图像宽度调整为15个像素点
%     imagesc(f_sampled, t, img_resized); % 用采样频率绘制图像
%     imagesc(f_sampled, t, img); % 用采样频率绘制图像
%     caxis([min_value max_value])
%     colormap(jet); % 设置当前颜色图样式
% 
%     % 调整图形窗口的尺寸和分辨率
%     set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [846.6, 340.2, 120, 480]);
%     set(gca, 'linewidth', 1.5, 'fontsize', 12, 'fontname', 'Times New Roman');
% 
%     set(gca, 'YTick', 0:25:129);
%     y_label = [];
%     set(gca, 'yticklabel', y_label);
%     set(gca, 'YDir', 'reverse');
%     
%     x_label = [];
%     set(gca, 'xticklabel', x_label);
%     
%     set(gcf, 'Color', 'w');
%     text(-22, 250, '(d)', 'horiz', 'left', 'fontsize', 12, 'fontname', 'Times New Roman');
%     set(gca, 'Visible', 'off') % 去除边框
%     h1 = gca;
%     RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
%     set(gca, 'Visible', 'off') % 去除边框
%     
%     file_path1 = 'D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\测试窗口数据集1\CWT_Trace_block\';
%     % 检查文件夹是否存在，不存在则创建
%     if ~exist(file_path1, 'dir')
%         mkdir(file_path1);
%     end
%     filenm = ['CWT_' num2str(i) '.bmp'];
%     save_path = strcat(file_path1, filenm);
%     print(gcf, '-dtiff', '-r50', save_path); % 用图片本来名称保存

    %% 滑窗大小
    for block = 5:5:30
        
        % 定义文件夹命名
        folder_name = sprintf('CWT_%03d_%d', i, block); % 文件夹命名格式
        file_path2 = fullfile('D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\测试窗口数据集1\CWT\', folder_name);

        % 检查文件夹是否存在，不存在则创建
        if ~exist(file_path2, 'dir')
            mkdir(file_path2);
        end
        
        %% 切块    
        for j = 1:1:119       
            
            %% 时频起始位置
            d_start = (30-block)/2 + j;

            %% 时频谱切块
            figure
            img = abs(coefs(idx_sampled, :))';
            img_resized = imresize(img, [size(img, 1), 15]); % 将图像宽度调整为15个像素点
            block_spec = img_resized(d_start:d_start+block, :); % 使用滑窗切块数据
            imagesc(f_sampled, d_start:d_start+block, block_spec); % 用采样频率绘制图像
            caxis([min_value max_value])
            colormap(jet); % 设置当前颜色图样式 
            set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 15/50 2*block/100]); % 设置图形窗口大小
            set(gca, 'linewidth', 1.5, 'fontsize', 12, 'fontname', 'Times New Roman');
            set(gca, 'YTick', 0:25:129);
            y_label = [];
            set(gca, 'yticklabel', y_label);
            set(gca, 'YDir', 'reverse');
            x_label = [];
            set(gca, 'xticklabel', x_label);
            set(gcf, 'Color', 'w');
            text(-22, 250, '(d)', 'horiz', 'left', 'fontsize', 12, 'fontname', 'Times New Roman');
            set(gca, 'Visible', 'off') % 去除边框
            h1 = gca;
            RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
            set(gca, 'Visible', 'off') % 去除边框
            % 图片命名和保存
            filenm = sprintf('CWT_%03d_%03d.bmp', i, j); % 图片命名格式
            save_path = fullfile(file_path2, filenm);
            print(gcf, '-dtiff', '-r50', save_path); % 使用图片本来的名称保存
            close all;
        end
    end
end
