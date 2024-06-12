clear all;
close all;
clc;
%%%%%%%%%%%%%��������%%%%%%%%%%%%%%%%%%%%%
[seismic, sampint, ch, bh,th]=altreadsegy('copy_post_stack_300_256_delta30.sgy','textHeader','yes',...
    'textformat','ascii','fpformat','ibm','binaryheader','yes','traceheaders','yes');

% ntrace=400;   %�������ݵĵ��� 
% nsample=sampint; %�������ݲ�������
[nsample,ntrace]=size(seismic);

% dts=0.002;   %�������ݲ����ʡ���λs
dt=0.004;
fs=1/dt;
t = 0:1:nsample-1;

%% ISD����
fstart = 1;%��λ�±߽�
fstop  = 120;%��λ�ϱ߽�
delpha =  1;%Ƶ��ɨ����
f_num = fstart:delpha:fstop;
WaveMat = zeros(nsample,nsample*length(f_num));
%% �����Ӳ�����
for k = 1:1:length(f_num)
    k
    source = ricker(dt,f_num(k));
    source =hilbert(source);
    wvletMatrix1=eye(nsample);
    wvletMatrix=conv2(wvletMatrix1,source,'same');
    WaveMat(:,(k-1)*nsample+1:k*nsample)=wvletMatrix;
end


% �����ض�����
%block = 29;
f_max = 80;
f_min = 10;
interval =5;

% ����Ƶ������
f = (0:nsample-1) * fs / nsample;
f = f(1:floor(nsample/2) + 1); % ֻȡ��Ƶ����

% �ҵ�Ƶ�ʷ�Χ��Ӧ������
f_sampled = f_min:interval:f_max;
idx_sampled = arrayfun(@(freq) find(abs(f - freq) == min(abs(f - freq)), 1), f_sampled);

well_trace = [63 64 65  240 241 242];% ����Ҫ�����ĵ���
for i= well_trace
    waitbar(i/ntrace);

    x = seismic(:,i);
    x = x/max(abs(x));

    %% L1L2���ʱƵ��
    pm.lambda = 0.25;
    pm.delta = 15000;
    pm.alpha = 1;
    pm.maxit = 100;
    [xADMM,X_fista] = CS_L1L2_uncon_ADMM(WaveMat,x,pm);

    Spec = reshape(xADMM,nsample,length(f_num));

    %% ����ʱƵ�׷�Χ
    max_value = max(max(abs(Spec)));
    min_value = min(min(abs(Spec)));

 %% ����ʱƵ��
    figure
    imagesc(f_min:f_max,t,abs(Spec(:,idx_min:idx_max)));% ��ʾǰ120Ƶ�ʵ�ʱƵ��

    caxis([min_value max_value])% ������ɫ�᷶Χ
    JET = colormap(jet);% ʹ��JETɫͼ
    colormap (JET);   % Set the current colormap Style 

    % ����ͼ������
    set(gcf,'position',[846.6,340.2,120,480]);
    set(gca,'linewidth',1.5,'fontsize',12,'fontname','Times New Roman');

    set(gca,'YTick',0:25:129);
    %y_label = 3.2:0.1:3.7;  %%%%%ת��Ϊ����
    y_label = [];
    set(gca,'yticklabel',y_label);
    set(gca,'YDir','reverse'); % ��תY��
    %ylabel('Time(s)','FontSize',12);

    %set(gca,'XTick',10:70:80);
    x_label = [];
    set(gca,'xticklabel',x_label);
    %xlabel('Frequency(Hz)','FontSize',12);

    set(gcf,'Color','w'); % ���ñ���ɫΪ��ɫ
    text(-22,250,'(d)','horiz','left','fontsize',12,'fontname','Times New Roman');
    %title('CWT(\sl{f_{b}} \rm= 0.5, \sl{f_{c}} \rm= 1)','fontsize',12,'fontname','Times New Roman');
    %title('CWT','fontsize',12,'fontname','Times New Roman');
    h1 = gca;
    RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
    set(gca,'Visible','off') %ȥ���߿� ����������


    file_path1 =  'D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\���Դ������ݼ�3\ISD_Trace_block\';
    % ����ļ����Ƿ���ڣ��������򴴽�
    if ~exist(file_path1, 'dir')
        mkdir(file_path1);
    end
    filenm = ['ISD_' num2str(i) '.bmp' ];
    save_path = strcat(file_path1,filenm);
    print(gcf,'-dtiff','-r50',save_path);%%��ͼƬ�������Ʊ���
    
    close(gcf);
    
    %% ������С
    for block = 5:5:30
        % �����ļ�������
        folder_name = sprintf('ISD_%03d_%d', i, block); % �ļ���������ʽ
        file_path2 = fullfile('D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\���Դ������ݼ�3\ISD\', folder_name);

        % ����ļ����Ƿ���ڣ��������򴴽�
        if ~exist(file_path2, 'dir')
            mkdir(file_path2);
        end
        
        %% �п�    
        for j = 1:1:119       

           %% ʱƵ��ʼλ��
            d_start = (30-block)/2 + j;

           %% ʱƵ���п�
            figure;
            img = abs(Spec(:, idx_sampled));
            img_resized = imresize(img, [size(img, 1), 15]); % ��ͼ���ȵ���Ϊ15�����ص�
            block_spec = img_resized(d_start:d_start+block, :); % ʹ�û����п�����
            imagesc(f_sampled, d_start:d_start+block, block_spec); % �ò���Ƶ�ʻ���ͼ��
            
            caxis([min_value max_value])
            JET = colormap(jet);
            colormap (JET);   % Set the current colormap Style 
            set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 15/50 2*block/100]); % ����ͼ�δ��ڴ�С
            set(gca,'linewidth',1.5,'fontsize',12,'fontname','Times New Roman');
            set(gca,'YTick',0:25:129);
            %y_label = 3.2:0.1:3.7;  %%%%%ת��Ϊ����
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
            set(gca,'Visible','off') %ȥ���߿� 
            % ͼƬ�����ͱ���
            filenm = sprintf('ISD_%03d_%03d.bmp', i, j); % ͼƬ������ʽ
            save_path = fullfile(file_path2, filenm);
            print(gcf, '-dtiff', '-r50', save_path); % ʹ��ͼƬ���������Ʊ���
            close all;
        end
    end
end

