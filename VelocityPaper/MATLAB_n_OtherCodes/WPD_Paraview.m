%% WPD calculations:
clear all
clc
%%
path1 = 'ANN/15ang/';
path2 = 'DATASET/SmallBox/';
angle = 15;
ML = readmatrix([path1 ['UxUyUz_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
TurbInts = readmatrix([path1 ['IuIwIv_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
It = readmatrix([path1 ['It_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
Les = importdata([path2 ['LES_Output_VelStudy_' num2str(angle) '_SB' '.txt']]);
LES = Les.data;
LES_UMean = LES(:,[1, 2:4]);
%%
UxUyUz = ML(:,2:4);

UMag = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,3).^2); % + UxUyUz(:,2).^2 + UxUyUz(:,3).^2

Umag_z = sqrt(UxUyUz(:,1).^2);

Uh = [12.9943, 12.9943, 12.9911, 12.9954, 13.0051, 13.0051, 13.0388];

WPD_Norm_z = Umag_z.^3./Uh(:,3)^3;

WPD_Norm = UMag.^3./Uh(:,3)^3;

Diff = WPD_Norm - WPD_Norm_z;
%%
WPD_It = [It(:,1),WPD_Norm,It(:,2)];

WPD_l1_temp = WPD_It(:,2) < 1;
WPD_l1 = WPD_It(WPD_l1_temp,:);
c05 = 0.5*ones(size(WPD_l1,1),1);
WPD_l1_05 = [WPD_l1 c05];

WPD_g1_gIt18_temp = WPD_It(:,2) > 1 & WPD_It(:,3) > 18;
WPD_g1_gIt18 = WPD_It(WPD_g1_gIt18_temp,:);
c15 = 1.5*ones(size(WPD_g1_gIt18,1),1);
WPD_g1_gIt18_15 = [WPD_g1_gIt18 c15];

WPD_g1_l15_lIt18_temp = WPD_It(:,2) > 1 & WPD_It(:,2) < 1.5 & WPD_It(:,3) < 18;
WPD_g1_l15_lIt18 = WPD_It(WPD_g1_l15_lIt18_temp,:);
c25 = 2.5*ones(size(WPD_g1_l15_lIt18,1),1);
WPD_g1_l15_lIt18_25 = [WPD_g1_l15_lIt18 c25];

WPD_g15_l25_lIt18_temp = WPD_It(:,2) > 1.5 & WPD_It(:,2) < 2.5 & WPD_It(:,3) < 18;
WPD_g15_l25_lIt18 = WPD_It(WPD_g15_l25_lIt18_temp,:);
c35 = 3.5*ones(size(WPD_g15_l25_lIt18,1),1);
WPD_g15_l25_lIt18_35 = [WPD_g15_l25_lIt18 c35];

WPD_g25_lIt18_temp = WPD_It(:,2) > 2.5 & WPD_It(:,3) < 18;
WPD_g25_lIt18 = WPD_It(WPD_g25_lIt18_temp,:);
c45 = 4.5*ones(size(WPD_g25_lIt18,1),1);
WPD_g25_lIt18_45 = [WPD_g25_lIt18 c45];

SizeTotal = size(WPD_l1_05,1) + size(WPD_g1_gIt18_15,1) + size(WPD_g1_l15_lIt18_25,1) + size(WPD_g15_l25_lIt18_35,1) + size(WPD_g25_lIt18_45,1);

WPD_all = [WPD_l1_05;WPD_g1_gIt18_15;WPD_g1_l15_lIt18_25;WPD_g15_l25_lIt18_35;WPD_g25_lIt18_45];

% PerZone0 = size(WPD_l1_05,1)/SizeTotal
% PerZone1 = size(WPD_g1_gIt18_15,1)/SizeTotal
% PerZone2 = size(WPD_g1_l15_lIt18_25,1)/SizeTotal
PerZone3 = (size(WPD_g15_l25_lIt18_35,1)/SizeTotal)*100
PerZone4 = (size(WPD_g25_lIt18_45,1)/SizeTotal)*100

% TPerc = PerZone0 + PerZone1 + PerZone2 + PerZone3 + PerZone4

for i = 1:length(WPD_all(:,1))
    ID = find(LES_UMean(i,1) == WPD_all(:,1));
    Row(i,:) = WPD_all(ID,:);
end

Temp = array2table(Row(:,[1,4]), 'VariableNames', {'cellID','It'}); %'cellID','Iu','Iw','Iv'
a = ['WPD_VAWT_HM_OB_OFS_' num2str(angle) '_vtk' '_v2' '.csv'];
writetable(Temp,[path1, a],'Delimiter',',')
%%
WPD_weak_Uz = [LES_UMean(:,1),WPD_Norm];

Temp = array2table(WPD_weak_Uz, 'VariableNames', {'cellID','WPD'}); %'cellID','Iu','Iw','Iv'
a = ['WPD_VAWT_HM_OB_OFS_' num2str(angle) '_vtk' '_values' '.csv'];
writetable(Temp,[path1, a],'Delimiter',',')
%%


% [MaxWPD, IdMax] = max(WPD_Norm);
% 
% MaxTurbInts = TurbInts(IdMax, :);
% MaxIt = It(IdMax, :);
% HeightMax = LES_UMean(IdMax, 3) - 0.4;