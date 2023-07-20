%% WPD calculations:
clear all
clc
%%
% path1 = 'ANN/45/';
% path2 = 'DATASET/SmallBox/';
% angle = 45;
% ML = readmatrix([path1 ['UxUyUz_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
% TurbInts = readmatrix([path1 ['IuIwIv_41OB_OFS_' num2str(angle) '_vtk' '.csv']]);
% It = readmatrix([path1 ['It_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
% Les = importdata([path2 ['LES_Output_VelStudy_' num2str(angle) '_SB' '.txt']]);
% LES = Les.data;
% LES_UMean = LES(:,[1, 2:4]);
% %%
% UxUyUz = ML(:,2:4);
% 
% UMag = sqrt(UxUyUz(:,1).^2); % + UxUyUz(:,2).^2 + UxUyUz(:,3).^2
% 
% Uh = [12.9943, 12.9943, 12.9911, 12.9954, 13.0051, 13.0051, 13.0388];
% 
% WPD_Norm = UMag.^3./Uh(:,7)^3;
% 
% [MaxWPD, IdMax] = max(WPD_Norm);
% 
% MaxTurbInts = TurbInts(IdMax, :);
% MaxIt = It(IdMax, :);
% HeightMax = LES_UMean(IdMax, 3) - 0.4;
% %%
% UMag_XY = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,2).^2);
% WPD_Norm_XY = UMag_XY.^3./Uh(:,1)^3;
% 
% [MaxWPD_XY, IdMax_XY] = max(WPD_Norm_XY);
% 
% MaxIt_XY = It(IdMax_XY, :);
% HeightMax_XY = LES_UMean(IdMax_XY, 3) - 0.4;
% %%
% UMag_XZ = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,3).^2);
% WPD_Norm_XZ = UMag_XZ.^3./Uh(:,1)^3;
% 
% [MaxWPD_XZ, IdMax_XZ] = max(WPD_Norm_XZ);
% 
% MaxIt_XZ = It(IdMax_XZ, :);
% HeightMax_XZ = LES_UMean(IdMax_XZ, 3) - 0.4;
%%
Angles = [0,75,15,225,30,375,45]; %
PathAngles = {'ANN/0_weak/','ANN/75/','ANN/15ang/','ANN/225/','ANN/30/','ANN/375/','ANN/45/'};

% Angles = [15]; %
% PathAngles = {'ANN/15ang_weak/'};
% % 
% PathAngles = {'ANN/0/ForError/','ANN/75/ForError/','ANN/15ang/ForError/',...
%     'ANN/225/ForError/','ANN/30/ForError/','ANN/375/ForError/','ANN/45/ForError/'};

for i = 1:length(Angles)
    path1 = PathAngles{i};
    path2 = 'DATASET/SmallBox/';
%     angle = 45;
    angle = Angles(i);
    ML = readmatrix([path1 ['UxUyUz_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
    TurbInts = readmatrix([path1 ['IuIwIv_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
    It = readmatrix([path1 ['It_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
    IT_angs{i} = It;
    Les = importdata([path2 ['LES_Output_VelStudy_' num2str(angle) '_SB' '.txt']]);
    LES = Les.data;
    LES_UMean = LES(:,[1, 2:4]);
%     UxUyUz = ML(:,2:4);
    UxUyUz = LES(:,5:7);
    UxUyUz_LES{i} = LES(:,5:7);
    UxUyUz_ML{i} = ML(:,2:4);
    UMag = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,3).^2); % + UxUyUz(:,2).^2 + UxUyUz(:,3).^2

    Uh = [12.9943, 12.9943, 12.9911, 12.9954, 13.0051, 13.0051, 13.0388];
    
    %%
    WPD_Norm{i} = UMag.^3./Uh(:,i)^3;
    
    [MaxWPD(:,i), IdMax(:,i)] = maxk(WPD_Norm{i},1);

    MaxTurbInts = TurbInts(IdMax(1,i), :);
    LocImp(i,:) = LES_UMean(IdMax(1,i),:);
    MinIt(:,i) = It(IdMax(1,i), :);
    HeightMax(:,i) = LES_UMean(IdMax(1,i), 3) - 0.4;
    
    %%
    UMag_XY = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,2).^2);
    WPD_Norm_XY{i} = UMag_XY.^3./Uh(:,i)^3;

    [MaxWPD_XY(:,i), IdMax_XY(:,i)] = maxk(WPD_Norm_XY{i},1);

    MinIt_XY(:,i) = It(IdMax_XY(1,i), :);
    
    LocImp_XY(i,:) = LES_UMean(IdMax_XY(1,i),:);
    HeightMax_XY(:,i) = LES_UMean(IdMax_XY(1,i), 3) - 0.4;  
    
    TempHeights(:,i) = LES_UMean(IdMax_XY(:,i),3);
    
    %%
    UMag_XZ = sqrt(UxUyUz(:,1).^2 + UxUyUz(:,3).^2);
    WPD_Norm_XZ{i} = UMag_XZ.^3./Uh(:,i)^3;

    [MaxWPD_XZ(:,i), IdMax_XZ(:,i)] = max(WPD_Norm_XZ{i});

    MinIt_XZ(:,i) = It(IdMax_XZ(:,i), :);
    HeightMax_XZ(:,i) = LES_UMean(IdMax_XZ(:,i), 3) - 0.4;   
    
    LocImp_XZ(i,:) = LES_UMean(IdMax_XZ(1,i),:);
%     HeightMax_XZ(:,i) = LES_UMean(IdMax_XY(1,i), 3) - 0.4;  
    
    TempHeights_XZ(:,i) = LES_UMean(IdMax_XZ(:,i),3);
    
    
end
% %%
% CellIDs = LES_UMean(IdMax(:,1),:);
% 
% Temp = find(abs(WPD_Norm{1}(:,1) - 2.8135) < 1*10^-3)
% NewCellWPD = WPD_Norm{1}(Temp,1);
% NewCell = LES_UMean(Temp,:);
% 
% Coords_0 = NewCell(:,2:end);
% Coords_0_mirr = [Coords_0(:,1), Coords_0(:,2), -Coords_0(:,3)];
% 
% Temp2 = find(abs(LES_UMean(:,2) - Coords_0_mirr(:,1)) < 3*10^-3 & (LES_UMean(:,3) - Coords_0_mirr(:,2)) < 3*10^-3 & abs(LES_UMean(:,4) - Coords_0_mirr(:,3)) < 3*10^-3)
% NewCell_2 = LES_UMean(Temp2,:);
% NewCellWPD_2 = WPD_Norm{1}(Temp2,1);
% CorrIt_newCell = IT_angs{1}(Temp2,:);
% 
% NewCellWPD_2_XZ = WPD_Norm_XZ{1}(Temp2,1);
% CorrIt_newCell_XZ = IT_angs{1}(Temp2,:); 435481000000000
%% Line 1
% Temp_vawt_line = find(abs(LES_UMean(:,2) - LocImp_XY(2,2)) < 3*10^-3 & abs(LES_UMean(:,3) - LocImp_XY(2,3)) < 3*10^-3);
% Temp_vawt_line = find(abs(LES_UMean(:,2) - LocImp_XZ(2,2)) < 3*10^-3 & abs(LES_UMean(:,3) - LocImp_XZ(2,3)) < 3*10^-3);
Temp_vawt_line = find(abs(LES_UMean(:,4) - (-0.04)) < 3*10^-3 & abs(LES_UMean(:,3) - 0.437) < 3*10^-3);

% Temp_vawt_line = find(abs(LES_UMean(:,3) - LocImp_XY(2,3)) < 3*10^-3 & abs(LES_UMean(:,2) - (-0.003)) < 2.5*10^-3);

Vawt_line = LES_UMean(Temp_vawt_line,:);

for jjj = 1:7
    WPD_HAWT_LINE(:,jjj) = WPD_Norm{jjj}(Temp_vawt_line,:);
    WPD_VAWT_VM_LINE(:,jjj) = WPD_Norm_XY{jjj}(Temp_vawt_line,:);
    WPD_VAWT_HM_LINE(:,jjj) = WPD_Norm_XZ{jjj}(Temp_vawt_line,:);
end

WPD_HAWT_LINE_1 = [WPD_HAWT_LINE, Vawt_line(:,2)];
WPD_HAWT_LINE_2 = sortrows(WPD_HAWT_LINE_1,8);

TempSort_WPD_HAWT = [WPD_HAWT_LINE, Vawt_line(:,1:2)];
Sort_WPD_HAWT = sortrows(TempSort_WPD_HAWT,9);

WPD_VAWT_VM_LINE_1 = [WPD_VAWT_VM_LINE, Vawt_line(:,2)];
WPD_VAWT_VM_LINE_2 = sortrows(WPD_VAWT_VM_LINE_1,8);

WPD_VAWT_HM_LINE_1 = [WPD_VAWT_HM_LINE, Vawt_line(:,2)];
WPD_VAWT_HM_LINE_2 = sortrows(WPD_VAWT_HM_LINE_1,8);
ANGs = [0,7.5,15,22.5,30,37.5,45];
UP = 0.0667;
Down = -0.067;
Div = (UP - Down)/23;
% Length = Down:Div:UP;
% path4 = 'WPDPlots/';
% Combo = [WPD_HAWT_LINE_2(:,end), WPD_HAWT_LINE_2(:,end-1) WPD_VAWT_VM_LINE_2(:,end-1) WPD_VAWT_HM_LINE_2(:,end-1)];
Combo = [WPD_HAWT_LINE_2(:,end) WPD_HAWT_LINE_2(:,3) WPD_VAWT_VM_LINE_2(:,3) WPD_VAWT_HM_LINE_2(:,3)];
%%
S_WPD_HAWT_LINE_2 = smooth(WPD_HAWT_LINE_2(:,3));
S_WPD_VAWT_VM_LINE_2 = smooth(WPD_VAWT_VM_LINE_2(:,3));
S_WPD_VAWT_HM_LINE_2 = smooth(WPD_VAWT_HM_LINE_2(:,3));

Combo_S = [WPD_HAWT_LINE_2(:,end) S_WPD_HAWT_LINE_2 S_WPD_VAWT_VM_LINE_2 S_WPD_VAWT_HM_LINE_2];
%%
WPD_HAWT_LINE_1_temp = [WPD_HAWT_LINE, Vawt_line(:,[1,end])];
WPD_HAWT_LINE_2_temp = sortrows(WPD_HAWT_LINE_1_temp,9);

%% LES WPD

UMag_XZ = sqrt(UxUyUz_LES{6}(:,1).^2 + UxUyUz_LES{6}(:,3).^2);

WPD_LES_XZ = UMag_XZ.^3/13.0051^3;

%%

LES_val = LES(Temp_vawt_line,:);


% Temp = array2table(Combo, 'VariableNames', {'Length', 'WPD_HAWT','WPD_VAWT_VM', 'WPD_VAWT_HM'});
% a = ['WPD_VAWT_LineComp_2' '.csv'];
% writetable(Temp,[path4, a],'Delimiter',',')


%% TI
% for jjj = 1:7
%     WPD_HAWT_LINE(:,jjj) = WPD_Norm{jjj}(Temp_vawt_line,:);
%     WPD_VAWT_VM_LINE(:,jjj) = WPD_Norm_XY{jjj}(Temp_vawt_line,:);
%     WPD_VAWT_HM_LINE(:,jjj) = WPD_Norm_XZ{jjj}(Temp_vawt_line,:);
% end
% 
% WPD_HAWT_LINE_1 = [WPD_HAWT_LINE, Vawt_line(:,end)];
% WPD_HAWT_LINE_2 = sortrows(WPD_HAWT_LINE_1,8);
% 
% WPD_VAWT_VM_LINE_1 = [WPD_VAWT_VM_LINE, Vawt_line(:,end)];
% WPD_VAWT_VM_LINE_2 = sortrows(WPD_VAWT_VM_LINE_1,8);
% 
% WPD_VAWT_HM_LINE_1 = [WPD_VAWT_HM_LINE, Vawt_line(:,end)];
% WPD_VAWT_HM_LINE_2 = sortrows(WPD_VAWT_HM_LINE_1,8);
% ANGs = [0,7.5,15,22.5,30,37.5,45];
% UP = 0.0667;
% Down = -0.067;
% Div = (UP - Down)/23;
% % Length = Down:Div:UP;
% % path4 = 'WPDPlots/';
% Combo = [WPD_HAWT_LINE_2(:,end), WPD_HAWT_LINE_2(:,end-1) WPD_VAWT_VM_LINE_2(:,end-1) WPD_VAWT_HM_LINE_2(:,end-1)];

%% Line 2

TempFind = find(LES_UMean(:,1) == 167224);
LocImp2 = LES_UMean(TempFind,:);

Temp_vawt_line_2 = find(abs(LES_UMean(:,2) - LocImp2(:,2)) < 3*10^-3 & abs(LES_UMean(:,3) - LocImp2(:,3)) < 3*10^-3);
Vawt_line_2 = LES_UMean(Temp_vawt_line_2,:);

for jjj = 1:7
    WPD_HAWT_LINE2(:,jjj) = WPD_Norm{jjj}(Temp_vawt_line_2,:);
    WPD_VAWT_VM_LINE2(:,jjj) = WPD_Norm_XY{jjj}(Temp_vawt_line_2,:);
    WPD_VAWT_HM_LINE2(:,jjj) = WPD_Norm_XZ{jjj}(Temp_vawt_line_2,:);
end

WPD_HAWT_LINE_12 = [WPD_HAWT_LINE2, Vawt_line_2(:,end)];
WPD_HAWT_LINE_22 = sortrows(WPD_HAWT_LINE_12,8);

WPD_VAWT_VM_LINE_12 = [WPD_VAWT_VM_LINE2, Vawt_line_2(:,end)];
WPD_VAWT_VM_LINE_22 = sortrows(WPD_VAWT_VM_LINE_12,8);

WPD_VAWT_HM_LINE_12 = [WPD_VAWT_HM_LINE2, Vawt_line_2(:,end)];
WPD_VAWT_HM_LINE_22 = sortrows(WPD_VAWT_HM_LINE_12,8);
ANGs = [0,7.5,15,22.5,30,37.5,45];

% path4 = 'WPDPlots/';
% Combo2 = [WPD_HAWT_LINE_22(:,end), WPD_HAWT_LINE_22(:,end-1) WPD_VAWT_VM_LINE_22(:,end-1) WPD_VAWT_HM_LINE_22(:,end-1)];
% Temp2 = array2table(Combo2, 'VariableNames', {'Length', 'WPD_HAWT','WPD_VAWT_VM', 'WPD_VAWT_HM'});
% a = ['WPD_VAWT_LineComp_2' '.csv'];
% writetable(Temp2,[path4, a],'Delimiter',',')

% path1 = 'ANN/0/';
% angle = 0;
% ML = readmatrix([path1 ['UxUyUz_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
% TurbInts = readmatrix([path1 ['IuIwIv_OB_OFS_' num2str(angle) '_vtk' '.csv']]);
% It = readmatrix([path1 ['It_OB_OFS_' num2str(angle) '_vtk' '.csv']]);

% Idx1 = find(TurbInts(:,end) >100);
% Idx2 = find(TurbInts(:,end-1) >100);
% Idx3 = find(TurbInts(:,end-2) >100);
% Idx4 = find(It(:,end) >100);
% Idx_all = [Idx1;Idx2;Idx3;Idx4];
% Idx = unique(Idx_all);
% 
% ML_OIt = ML(setdiff(1:end,Idx),:);
% It_OIt = It(setdiff(1:end,Idx),:);
% TurbInts_OIt = TurbInts(setdiff(1:end,Idx),:);
% 
% UxUyUz_OIt = ML_OIt(:,2:4);
% 
% UMag_OIt = sqrt(UxUyUz_OIt(:,1).^2 + UxUyUz_OIt(:,2).^2 + UxUyUz_OIt(:,3).^2);
% 
% Uh = [12.9943, 12.9943, 12.9911, 12.9954, 13.0051, 13.0051, 13.0388];
% 
% WPD_Norm_OIt = UMag_OIt.^3./Uh(:,1)^3;
% 
% [MaxWPD_OIt, IdMax_OIt] = max(WPD_Norm_OIt);
% 
% MaxTurbInts_OIt = TurbInts_OIt(IdMax_OIt, :);
% MaxIt_OIt = It_OIt(IdMax_OIt, :);
