%%
clc
clear all
close all

make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.05 0.05], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

path1 = 'Dataset2/SB_RansIt/';
path7 = 'Dataset2/SB_RansIt_org/';
path2 = 'DATASET/SB/';
path4 = 'ANN/15/OB/Figures_comb/'; %15_a0_a225_a45  ANN/15/OB/Figures_wRans/

path5 = 'ANN/15/OB/'; % 'ANN/15/OB/'
path6 = 'VelocityLines/';
path8 = 'ANN/15/OB/ResultFiles_SO/'; %15_a0_a225_a45  ANN/15/OB/Figures_wRans/

Files = {'Iu_OB', 'Iw_OB', 'Iv_OB', 'It_OB', 'Ux_mean_OB', 'Uy_mean_OB', 'Uz_mean_OB'};
Legends1 = {'$I_{u_\mathrm{LES}}$', '$I_{w_\mathrm{LES}}$', '$I_{v_\mathrm{LES}}$', '$I_{t_\mathrm{LES}}$', '$\bar{U_x}_\mathrm{LES}$', '$\bar{U_y}_\mathrm{LES}$', '$\bar{U_z}_\mathrm{LES}$'};
Legends2 = {'$I_{u_\mathrm{ML}}$', '$$I_{w_\mathrm{ML}}$$', '$$I_{v_\mathrm{ML}}$', '$$I_{t_\mathrm{ML}}$', '$\bar{U_x}_\mathrm{ML}$', '$\bar{U_y}_\mathrm{ML}$', '$\bar{U_z}_\mathrm{ML}$'};
Legends3 = {'$I_{u_\mathrm{RANS}}$', '$I_{w_\mathrm{RANS}}$', '$I_{v_\mathrm{RANS}}$', '$I_{t_\mathrm{RANS}}$', '$\bar{U_x}_\mathrm{RANS}$', '$\bar{U_y}_\mathrm{RANS}$', '$\bar{U_z}_\mathrm{RANS}$'};
% angle = 15;
tag0 = 'OB';
angle = 15;
tag1 = 'SO_K4';
Lines = {'p20','p36','p52'};
%%
for i = 7 %% Put 1:3 or 5:7 or just 4
    File = xlsread([path5 [Files{i} '_K4_sort' '.xlsx']]);
    for j = 1:3
        Line = csvread([path6 [Lines{j} '.csv']]);
        [C,idx_uq] = unique(Line(:,13));
        Lines_unq = Line(idx_uq,[13, 16:end]);

        id = ismember(File(:,1), Lines_unq(:,1));
        ML_roof1 = File(id,:);
    
        Les = importdata([path2 ['LES-a' num2str(angle) '_SB' '.txt']]);
        LES = Les.data;
    
        LES_0 = find(LES(:,1) == Line(1,13));
        LES_1 = find(LES(:,1) == Line(16,13));
        LES_2 = find(LES(:,1) == Line(34,13));
        LES_zero = [LES_0, LES_1, LES_2];
        LES_start = LES(LES_zero,:);
        Temp =LES_start(:,[1:4, i, i]); %%%%% for velocities i ,i and It 11, 11 and for Iu, Iv, Iw i+7, i+7
        ML_roof2 = [Temp; ML_roof1];
%         ML_roof2 = ML_roof1;
%     ML_roof2 = ML_roof1
        Rans = importdata([path1 ['RANS-a' num2str(angle) '_SB' '.txt']]);
        RANS = Rans.data;
    
        Rans_Org = importdata([path7 ['RANS-a' num2str(angle) '_SB' '.txt']]);
        RANS_Org = Rans_Org.data;
    
        RANS_roof1 = RANS(id,:);
    
        RANS_0 = find(RANS_Org(:,1) == Line(1,13));
        RANS_1 = find(RANS_Org(:,1) == Line(16,13));
        RANS_2 = find(RANS_Org(:,1) == Line(34,13));
        RANS_zero = [RANS_0, RANS_1, RANS_2];
    
        RANS_start = RANS_Org(RANS_zero,:);
        Temp2 =RANS_start(:,:);
        RANS_roof2 = [Temp2; RANS_roof1(:,1:end)];
    
        RANS_roof = sortrows(RANS_roof2, 3);

        ML_roof = sortrows(ML_roof2, 3);
    
        [~,idx] = unique(ML_roof(:,1),'stable');
        [~,idxRans] = unique(RANS_roof(:,1),'stable');
        ML_roof = ML_roof(idx,:);
        RANS_roof = RANS_roof(idxRans,:);
        
%         RANS_roof = RANS_roof(RANS_roof(:,end) <= 40, :); %%%% comment for velocities
%         ML_roof = ML_roof(ML_roof(:,end) <= 40, :);  %%%% comment for velocities
        arcLen1 = (ML_roof(:,3) - 0.4).*10; % ML_roof(:,3)./0.4;
%         arcLen1(1) = 0.02;
        arcLen = arcLen1;
       
        AL{j} = arcLen;
        ML{j} = ML_roof;
        RansRes{j} = RANS_roof;
        
%         Combo_Ivwu{j} = [AL{j} ML{j}(:,end-1:end)];
%         Temp = array2table(Combo_Ivwu{j}, 'VariableNames', {'ArcLen','ML','LES'});
%         a = [Files{i} '-a' num2str(angle) '_' tag0 '_' Lines{j} '_' tag1 '.csv'];
%         writetable(Temp,[path8, a],'Delimiter',',')
        Combo{j} = [AL{j} ML{j}(:,end-1:end)./12.9943 RansRes{j}(:,i)./12.9943];
        Temp = array2table(Combo{j}, 'VariableNames', {'ArcLen','ML','LES','RANS'});
        a = [Files{i} '-a' num2str(angle) '_' tag0 '_' Lines{j} '_' tag1 '.csv'];
        writetable(Temp,[path8, a],'Delimiter',',')
    end
end