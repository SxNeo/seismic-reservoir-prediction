clear all;
close all;
clc;
%%%%%%%%%%%%%地震数据%%%%%%%%%%%%%%%%%%%%%
[seismic, sampint, ch, bh,th]=altreadsegy('copy_post_stack_300_256_delta30.sgy','textHeader','yes',...
    'textformat','ascii','fpformat','ibm','binaryheader','yes','traceheaders','yes');

% ntrace=400;   %地震数据的道数 
% nsample=sampint; %地震数据采样点数
[nsample,ntrace]=size(seismic);

% dts=0.002;   %地震数据采样率。单位s
dt=0.004;
fs=1/dt;
t = 0:1:nsample-1;

%% ISD参数
fstart = 1;%相位下边界
fstop  = 120;%相位上边界
delpha =  1;%频率扫描间隔
f_num = fstart:delpha:fstop;
WaveMat = zeros(nsample,nsample*length(f_num));
%% 构建子波矩阵
for k = 1:1:length(f_num)
    k
    source = ricker(dt,f_num(k));
    source =hilbert(source);
    wvletMatrix1=eye(nsample);
    wvletMatrix=conv2(wvletMatrix1,source,'same');
    WaveMat(:,(k-1)*nsample+1:k*nsample)=wvletMatrix;
end


% 分析特定道集
%block = 29;
f_max = 80;
f_min = 10;
interval =5;

% 计算频率数组
f = (0:nsample-1) * fs / nsample;
f = f(1:floor(nsample/2) + 1); % 只取正频部分

% 找到频率范围对应的索引
f_sampled = f_min:interval:f_max;
idx_sampled = arrayfun(@(freq) find(abs(f - freq) == min(abs(f - freq)), 1), f_sampled);

well_trace = [63 64 65  240 241 242];% 定义要分析的道号
for i= well_trace
    waitbar(i/ntrace);

    x = seismic(:,i);
    x = x/max(abs(x));

    %% L1L2求解时频谱
    pm.lambda = 0.25;
    pm.delta = 15000;
    pm.alpha = 1;
    pm.maxit = 100;
    [xADMM,X_fista] = CS_L1L2_uncon_ADMM(WaveMat,x,pm);

    Spec = reshape(xADMM,nsample,length(f_num));

    %% 设置时频谱范围
    max_value = max(max(abs(Spec)));
    min_value = min(min(abs(Spec)));

 %% 单道时频谱
    figure
    imagesc(f_min:f_max,t,abs(Spec(:,idx_min:idx_max)));% 显示前120频率的时频谱

    caxis([min_value max_value])% 设置颜色轴范围
    JET = colormap(jet);% 使用JET色图
    colormap (JET);   % Set the current colormap Style 

    % 调整图形属性
    set(gcf,'position',[846.6,340.2,120,480]);
    set(gca,'linewidth',1.5,'fontsize',12,'fontname','Times New Roman');

    set(gca,'YTick',0:25:129);
    %y_label = 3.2:0.1:3.7;  %%%%%转化为毫秒
    y_label = [];
    set(gca,'yticklabel',y_label);
    set(gca,'YDir','reverse'); % 反转Y轴
    %ylabel('Time(s)','FontSize',12);

    %set(gca,'XTick',10:70:80);
    x_label = [];
    set(gca,'xticklabel',x_label);
    %xlabel('Frequency(Hz)','FontSize',12);

    set(gcf,'Color','w'); % 设置背景色为白色
    text(-22,250,'(d)','horiz','left','fontsize',12,'fontname','Times New Roman');
    %title('CWT(\sl{f_{b}} \rm= 0.5, \sl{f_{c}} \rm= 1)','fontsize',12,'fontname','Times New Roman');
    %title('CWT','fontsize',12,'fontname','Times New Roman');
    h1 = gca;
    RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
    set(gca,'Visible','off') %去除边框 隐藏坐标轴


    file_path1 =  'D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\测试窗口数据集3\ISD_Trace_block\';
    % 检查文件夹是否存在，不存在则创建
    if ~exist(file_path1, 'dir')
        mkdir(file_path1);
    end
    filenm = ['ISD_' num2str(i) '.bmp' ];
    save_path = strcat(file_path1,filenm);
    print(gcf,'-dtiff','-r50',save_path);%%用图片本来名称保存
    
    close(gcf);
    
    %% 滑窗大小
    for block = 5:5:30
        % 定义文件夹命名
        folder_name = sprintf('ISD_%03d_%d', i, block); % 文件夹命名格式
        file_path2 = fullfile('D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\测试窗口数据集3\ISD\', folder_name);

        % 检查文件夹是否存在，不存在则创建
        if ~exist(file_path2, 'dir')
            mkdir(file_path2);
        end
        
        %% 切块    
        for j = 1:1:119       

           %% 时频起始位置
            d_start = (30-block)/2 + j;

           %% 时频谱切块
            figure;
            img = abs(Spec(:, idx_sampled));
            img_resized = imresize(img, [size(img, 1), 15]); % 将图像宽度调整为15个像素点
            block_spec = img_resized(d_start:d_start+block, :); % 使用滑窗切块数据
            imagesc(f_sampled, d_start:d_start+block, block_spec); % 用采样频率绘制图像
            
            caxis([min_value max_value])
            JET = colormap(jet);
            colormap (JET);   % Set the current colormap Style 
            set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 15/50 2*block/100]); % 设置图形窗口大小
            set(gca,'linewidth',1.5,'fontsize',12,'fontname','Times New Roman');
            set(gca,'YTick',0:25:129);
            %y_label = 3.2:0.1:3.7;  %%%%%转化为毫秒
            y_label = [];
            set(gca,'yticklabel',y_label);
            set(gca,'YDir','reverse'); 
            %ylabel('Time(s)','FontSize',12);
            %set(gca,'XTick',10:70:80);
            x_label = [];
            set(gca,'xticklabel',x_label);
            %xlabel('Frequency(Hz)','FontSize',12);
            set(gcf,'Color','w');
            text(-22,250,'(d)','horiz','left','fontsize',12,'fontname','Times New Roman');
            %title('CWT(\sl{f_{b}} \rm= 0.5, \sl{f_{c}} \rm= 1)','fontsize',12,'fontname','Times New Roman');
            %title('CWT','fontsize',12,'fontname','Times New Roman');
            h1 = gca;
            RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
            set(gca,'Visible','off') %去除边框 
            % 图片命名和保存
            filenm = sprintf('ISD_%03d_%03d.bmp', i, j); % 图片命名格式
            save_path = fullfile(file_path2, filenm);
            print(gcf, '-dtiff', '-r50', save_path); % 使用图片本来的名称保存
            close all;
        end
    end
end

