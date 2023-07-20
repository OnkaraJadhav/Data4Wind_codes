%% Delete Outliners
%%
clc
clear all

WriteAng = [375];
path1 = 'Dataset2/LES/';
path2 = 'Dataset2/RANS/';
path3 = 'Dataset_1B/LES/';
path4 = 'Dataset_1B/RANS/';

em = [];

LES = cell(1,length(WriteAng));
RANS = cell(1,length(WriteAng));
for i = 1:length(WriteAng)
    wangle = WriteAng(i);
    les = importdata([path1 ['LES-a' num2str(wangle) '_BB' '.txt']]);
    tLES = les.data;
    Idx1 = find(tLES(:,2) >0.13333333);
    LES_new = tLES(setdiff(1:end,Idx1),:);
    LES{i} = LES_new;
    
    Rans = importdata([path2 ['RANS-a' num2str(wangle) '_BB' '.txt']]);
    tRans = Rans.data;
    Rans_new = tRans(setdiff(1:end,Idx1),:);
    RANS{i} = Rans_new;
end

Combo = [LES{1}]; %; LES{3} ; LES{3} '_a' num2str(WriteAng(2)) '_a' num2str(WriteAng(3))
Temp = array2table(Combo, 'VariableNames', {'cellID', 'X_coor', 'Y_coor', 'Z_coor', 'UMean_X' 'UMean_Y', 'UMean_Z', 'Iu', 'Iw', 'Uv', 'It'});
% Temp = array2table(Combo, 'VariableNames', {'cellID', 'X_coor', 'Y_coor', 'Z_coor', 'meanCp', 'CpPrime', 'Angle'});
a = ['LES-a' num2str(WriteAng(1)) '_OB' '.txt'];
% a = ['LES-a' num2str(WriteAng(1)) '_a' num2str(WriteAng(2)) '_a' num2str(WriteAng(3)) '_BB' '.txt'];
writetable(Temp,[path3, a],'Delimiter',' ')

Combo = [RANS{1}]; %  cellID X_coor Y_coor Z_coor UMean_X UMean_Y UMean_Z CpMean It Uin gradp  ; RANS{2}; RANS{3} '_a' num2str(WriteAng(2)) '_a' num2str(WriteAng(3))
Temp = array2table(Combo, 'VariableNames', {'cellID', 'X_coor', 'Y_coor', 'Z_coor', 'UMean_X', 'UMean_Y', 'UMean_Z', 'CpMean', 'It', 'Uin', 'gradp', 'Angle'});
a = ['RANS-a' num2str(WriteAng(1)) '_OB' '.txt'];
writetable(Temp,[path4, a],'Delimiter',' ')