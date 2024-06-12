clear all;
close all;
clc;

%%%%%%%%%%%%% ���ص������� %%%%%%%%%%%%%
[seismic, sampint, ch, bh, th] = altreadsegy('copy_post_stack_300_256_delta30.sgy', ...
    'textHeader', 'yes', 'textformat', 'ascii', 'fpformat', 'ibm', ...
    'binaryheader', 'yes', 'traceheaders', 'yes');

[nsample, ntrace] = size(seismic);

dt = 0.004; % �����ʣ��룩
fs = 1 / dt; % ����Ƶ��
t = 0:1:nsample-1;

%%%%%%%%%%%%% CWT ���� %%%%%%%%%%%%%
wavename = 'cmor0.2-2';
Fc = centfrq(wavename); % С��������Ƶ��
c = 2 * Fc * nsample;
scals = c ./ (1:nsample);
f = scal2frq(scals, wavename, 1 / fs); % ���߶�ת��ΪƵ��

f_max = 80;
f_min = 10;
interval = 5;

% �ҵ���ӽ����Ϊ5Hz��Ƶ�ʶ�Ӧ������
f_sampled = f_min:interval:f_max;
idx_sampled = arrayfun(@(freq) find(abs(f - freq) == min(abs(f - freq)), 1), f_sampled);

well_trace = [63 64 65 240 241 242];
for i = well_trace

    waitbar(i / ntrace);

    x = seismic(:, i);
    x = x / max(abs(x));

    coefs = cwt(x, scals, wavename); % ��������С��ϵ��

    %% ����ʱƵ�׷�Χ
    max_value = max(max(abs(coefs)));
    min_value = min(min(abs(coefs)));

    %% ����ʱƵ��
%     figure
%     img = abs(coefs(idx_sampled, :))';
%     img_resized = imresize(img, [size(img, 1), 63]); % ��ͼ���ȵ���Ϊ15�����ص�
%     imagesc(f_sampled, t, img_resized); % �ò���Ƶ�ʻ���ͼ��
%     imagesc(f_sampled, t, img); % �ò���Ƶ�ʻ���ͼ��
%     caxis([min_value max_value])
%     colormap(jet); % ���õ�ǰ��ɫͼ��ʽ
% 
%     % ����ͼ�δ��ڵĳߴ�ͷֱ���
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
%     set(gca, 'Visible', 'off') % ȥ���߿�
%     h1 = gca;
%     RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
%     set(gca, 'Visible', 'off') % ȥ���߿�
%     
%     file_path1 = 'D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\���Դ������ݼ�1\CWT_Trace_block\';
%     % ����ļ����Ƿ���ڣ��������򴴽�
%     if ~exist(file_path1, 'dir')
%         mkdir(file_path1);
%     end
%     filenm = ['CWT_' num2str(i) '.bmp'];
%     save_path = strcat(file_path1, filenm);
%     print(gcf, '-dtiff', '-r50', save_path); % ��ͼƬ�������Ʊ���

    %% ������С
    for block = 5:5:30
        
        % �����ļ�������
        folder_name = sprintf('CWT_%03d_%d', i, block); % �ļ���������ʽ
        file_path2 = fullfile('D:\1Inversion data\Ensemble-Learning\SAMME\data_pre\���Դ������ݼ�1\CWT\', folder_name);

        % ����ļ����Ƿ���ڣ��������򴴽�
        if ~exist(file_path2, 'dir')
            mkdir(file_path2);
        end
        
        %% �п�    
        for j = 1:1:119       
            
            %% ʱƵ��ʼλ��
            d_start = (30-block)/2 + j;

            %% ʱƵ���п�
            figure
            img = abs(coefs(idx_sampled, :))';
            img_resized = imresize(img, [size(img, 1), 15]); % ��ͼ���ȵ���Ϊ15�����ص�
            block_spec = img_resized(d_start:d_start+block, :); % ʹ�û����п�����
            imagesc(f_sampled, d_start:d_start+block, block_spec); % �ò���Ƶ�ʻ���ͼ��
            caxis([min_value max_value])
            colormap(jet); % ���õ�ǰ��ɫͼ��ʽ 
            set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 15/50 2*block/100]); % ����ͼ�δ��ڴ�С
            set(gca, 'linewidth', 1.5, 'fontsize', 12, 'fontname', 'Times New Roman');
            set(gca, 'YTick', 0:25:129);
            y_label = [];
            set(gca, 'yticklabel', y_label);
            set(gca, 'YDir', 'reverse');
            x_label = [];
            set(gca, 'xticklabel', x_label);
            set(gcf, 'Color', 'w');
            text(-22, 250, '(d)', 'horiz', 'left', 'fontsize', 12, 'fontname', 'Times New Roman');
            set(gca, 'Visible', 'off') % ȥ���߿�
            h1 = gca;
            RemoveSubplotWhiteArea(h1, 1, 1, 1, 1);
            set(gca, 'Visible', 'off') % ȥ���߿�
            % ͼƬ�����ͱ���
            filenm = sprintf('CWT_%03d_%03d.bmp', i, j); % ͼƬ������ʽ
            save_path = fullfile(file_path2, filenm);
            print(gcf, '-dtiff', '-r50', save_path); % ʹ��ͼƬ���������Ʊ���
            close all;
        end
    end
end
