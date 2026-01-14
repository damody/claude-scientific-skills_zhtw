# 資料匯入與匯出參考

## 目錄
1. [文字與 CSV 檔案](#文字與-csv-檔案)
2. [試算表](#試算表)
3. [MAT 檔案](#mat-檔案)
4. [影像](#影像)
5. [表格與資料類型](#表格與資料類型)
6. [低階檔案 I/O](#低階檔案-io)

## 文字與 CSV 檔案

### 讀取文字檔案

```matlab
% 建議使用的高階函數
T = readtable('data.csv');          % 以表格形式讀取（混合類型）
M = readmatrix('data.csv');         % 以數值矩陣形式讀取
C = readcell('data.csv');           % 以儲存格陣列形式讀取
S = readlines('data.txt');          % 以字串陣列形式讀取（按行）
str = fileread('data.txt');         % 將整個檔案讀為字串

% 帶選項讀取
T = readtable('data.csv', 'ReadVariableNames', true);
T = readtable('data.csv', 'Delimiter', ',');
T = readtable('data.csv', 'NumHeaderLines', 2);
M = readmatrix('data.csv', 'Range', 'B2:D100');

% 偵測匯入選項
opts = detectImportOptions('data.csv');
opts.VariableNames = {'Col1', 'Col2', 'Col3'};
opts.VariableTypes = {'double', 'string', 'double'};
opts.SelectedVariableNames = {'Col1', 'Col3'};
T = readtable('data.csv', opts);
```

### 寫入文字檔案

```matlab
% 高階函數
writetable(T, 'output.csv');
writematrix(M, 'output.csv');
writecell(C, 'output.csv');
writelines(S, 'output.txt');

% 帶選項寫入
writetable(T, 'output.csv', 'Delimiter', '\t');
writetable(T, 'output.csv', 'WriteVariableNames', false);
writematrix(M, 'output.csv', 'Delimiter', ',');
```

### Tab 分隔檔案

```matlab
% 讀取
T = readtable('data.tsv', 'Delimiter', '\t');
T = readtable('data.txt', 'FileType', 'text', 'Delimiter', '\t');

% 寫入
writetable(T, 'output.tsv', 'Delimiter', '\t');
writetable(T, 'output.txt', 'FileType', 'text', 'Delimiter', '\t');
```

## 試算表

### 讀取 Excel 檔案

```matlab
% 基本讀取
T = readtable('data.xlsx');
M = readmatrix('data.xlsx');
C = readcell('data.xlsx');

% 指定工作表
T = readtable('data.xlsx', 'Sheet', 'Sheet2');
T = readtable('data.xlsx', 'Sheet', 2);

% 指定範圍
M = readmatrix('data.xlsx', 'Range', 'B2:D100');
M = readmatrix('data.xlsx', 'Sheet', 2, 'Range', 'A1:F50');

% 帶選項讀取
opts = detectImportOptions('data.xlsx');
opts.Sheet = 'Data';
opts.DataRange = 'A2';
preview(opts.VariableNames)     % 檢查欄位名稱
T = readtable('data.xlsx', opts);

% 取得工作表名稱
[~, sheets] = xlsfinfo('data.xlsx');
```

### 寫入 Excel 檔案

```matlab
% 基本寫入
writetable(T, 'output.xlsx');
writematrix(M, 'output.xlsx');
writecell(C, 'output.xlsx');

% 指定工作表和範圍
writetable(T, 'output.xlsx', 'Sheet', 'Results');
writetable(T, 'output.xlsx', 'Sheet', 'Data', 'Range', 'B2');
writematrix(M, 'output.xlsx', 'Sheet', 2, 'Range', 'A1');

% 附加到現有工作表（使用 Range 指定起始位置）
writetable(T2, 'output.xlsx', 'Sheet', 'Data', 'WriteMode', 'append');
```

## MAT 檔案

### 儲存變數

```matlab
% 儲存工作區所有變數
save('data.mat');

% 儲存特定變數
save('data.mat', 'x', 'y', 'results');

% 帶選項儲存
save('data.mat', 'x', 'y', '-v7.3');    % 大檔案（>2GB）
save('data.mat', 'x', '-append');        % 附加到現有檔案
save('data.mat', '-struct', 's');        % 將結構體欄位儲存為變數

% 壓縮選項
save('data.mat', 'x', '-v7');            % 壓縮（預設）
save('data.mat', 'x', '-v6');            % 不壓縮，較快
```

### 載入變數

```matlab
% 載入所有變數
load('data.mat');

% 載入特定變數
load('data.mat', 'x', 'y');

% 載入到結構體
S = load('data.mat');
S = load('data.mat', 'x', 'y');
x = S.x;
y = S.y;

% 列出內容而不載入
whos('-file', 'data.mat');
vars = who('-file', 'data.mat');
```

### MAT 檔案物件（大檔案）

```matlab
% 建立 MAT 檔案物件以進行部分存取
m = matfile('data.mat');
m.Properties.Writable = true;

% 讀取部分資料
x = m.bigArray(1:100, :);       % 僅前 100 行

% 寫入部分資料
m.bigArray(1:100, :) = newData;

% 取得變數資訊
sz = size(m, 'bigArray');
```

## 影像

### 讀取影像

```matlab
% 讀取影像
img = imread('image.png');
img = imread('image.jpg');
img = imread('image.tiff');

% 取得影像資訊
info = imfinfo('image.png');
info.Width
info.Height
info.ColorType
info.BitDepth

% 讀取特定影格（多頁 TIFF、GIF）
img = imread('animation.gif', 3);  % 第 3 影格
[img, map] = imread('indexed.gif');  % 索引影像含色彩對應表
```

### 寫入影像

```matlab
% 寫入影像
imwrite(img, 'output.png');
imwrite(img, 'output.jpg');
imwrite(img, 'output.tiff');

% 帶選項寫入
imwrite(img, 'output.jpg', 'Quality', 95);
imwrite(img, 'output.png', 'BitDepth', 16);
imwrite(img, 'output.tiff', 'Compression', 'lzw');

% 寫入索引影像含色彩對應表
imwrite(X, map, 'indexed.gif');

% 附加到多頁 TIFF
imwrite(img1, 'multipage.tiff');
imwrite(img2, 'multipage.tiff', 'WriteMode', 'append');
```

### 影像格式

```matlab
% 支援的格式（部分列表）
% BMP  - Windows Bitmap
% GIF  - 圖形交換格式
% JPEG - 聯合圖像專家組
% PNG  - 可攜式網路圖形
% TIFF - 標籤圖像檔案格式
% PBM, PGM, PPM - 可攜式點陣圖格式

% 檢查支援的格式
formats = imformats;
```

## 表格與資料類型

### 建立表格

```matlab
% 從變數建立
T = table(var1, var2, var3);
T = table(var1, var2, 'VariableNames', {'Col1', 'Col2'});

% 從陣列建立
T = array2table(M);
T = array2table(M, 'VariableNames', {'A', 'B', 'C'});

% 從儲存格陣列建立
T = cell2table(C);
T = cell2table(C, 'VariableNames', {'Name', 'Value'});

% 從結構體建立
T = struct2table(S);
```

### 存取表格資料

```matlab
% 按變數名稱
col = T.VariableName;
col = T.('VariableName');
col = T{:, 'VariableName'};

% 按索引
row = T(5, :);              % 第 5 行
col = T(:, 3);              % 第 3 欄（作為表格）
data = T{:, 3};             % 第 3 欄（作為陣列）
subset = T(1:10, 2:4);      % 子集（作為表格）
data = T{1:10, 2:4};        % 子集（作為陣列）

% 邏輯索引
subset = T(T.Value > 5, :);
```

### 修改表格

```matlab
% 新增變數
T.NewVar = newData;
T = addvars(T, newData, 'NewName', 'Col4');
T = addvars(T, newData, 'Before', 'ExistingCol');

% 移除變數
T.OldVar = [];
T = removevars(T, 'OldVar');
T = removevars(T, {'Col1', 'Col2'});

% 重新命名變數
T = renamevars(T, 'OldName', 'NewName');
T.Properties.VariableNames{'OldName'} = 'NewName';

% 重新排列變數
T = movevars(T, 'Col3', 'Before', 'Col1');
T = T(:, {'Col2', 'Col1', 'Col3'});
```

### 表格操作

```matlab
% 排序
T = sortrows(T, 'Column');
T = sortrows(T, 'Column', 'descend');
T = sortrows(T, {'Col1', 'Col2'}, {'ascend', 'descend'});

% 唯一列
T = unique(T);
T = unique(T, 'rows');

% 連接表格
T = join(T1, T2);                   % 內連接（根據共同鍵）
T = join(T1, T2, 'Keys', 'ID');
T = innerjoin(T1, T2);
T = outerjoin(T1, T2);

% 堆疊/反堆疊
T = stack(T, {'Var1', 'Var2'});
T = unstack(T, 'Values', 'Keys');

% 分組操作
G = groupsummary(T, 'GroupVar', 'mean', 'ValueVar');
G = groupsummary(T, 'GroupVar', {'mean', 'std'}, 'ValueVar');
```

### 儲存格陣列

```matlab
% 建立儲存格陣列
C = {1, 'text', [1 2 3]};
C = cell(m, n);             % 空的 m×n 儲存格陣列

% 存取內容
contents = C{1, 2};         % 儲存格 (1,2) 的內容
subset = C(1:2, :);         % 儲存格子集（仍為儲存格陣列）

% 轉換
A = cell2mat(C);            % 轉為矩陣（如果相容）
T = cell2table(C);          % 轉為表格
S = cell2struct(C, fields); % 轉為結構體
```

### 結構體

```matlab
% 建立結構體
S.field1 = value1;
S.field2 = value2;
S = struct('field1', value1, 'field2', value2);

% 存取欄位
val = S.field1;
val = S.('field1');

% 欄位名稱
names = fieldnames(S);
tf = isfield(S, 'field1');

% 結構體陣列
S(1).name = 'Alice';
S(2).name = 'Bob';
names = {S.name};           % 擷取所有名稱
```

## 低階檔案 I/O

### 開啟與關閉檔案

```matlab
% 開啟檔案
fid = fopen('file.txt', 'r');   % 讀取
fid = fopen('file.txt', 'w');   % 寫入（覆寫）
fid = fopen('file.txt', 'a');   % 附加
fid = fopen('file.bin', 'rb');  % 讀取二進位
fid = fopen('file.bin', 'wb');  % 寫入二進位

% 檢查錯誤
if fid == -1
    error('無法開啟檔案');
end

% 關閉檔案
fclose(fid);
fclose('all');              % 關閉所有檔案
```

### 文字檔案 I/O

```matlab
% 讀取格式化資料
data = fscanf(fid, '%f');           % 讀取浮點數
data = fscanf(fid, '%f %f', [2 Inf]);  % 兩欄
C = textscan(fid, '%f %s %f');      % 混合類型

% 讀取行
line = fgetl(fid);          % 一行（無換行符）
line = fgets(fid);          % 一行（含換行符）

% 寫入格式化資料
fprintf(fid, '%d, %f, %s\n', intVal, floatVal, strVal);
fprintf(fid, '%6.2f\n', data);

% 讀取/寫入字串
str = fscanf(fid, '%s');
fprintf(fid, '%s', str);
```

### 二進位檔案 I/O

```matlab
% 讀取二進位資料
data = fread(fid, n, 'double');     % n 個 double
data = fread(fid, [m n], 'int32');  % m×n 個 int32
data = fread(fid, Inf, 'uint8');    % 所有位元組

% 寫入二進位資料
fwrite(fid, data, 'double');
fwrite(fid, data, 'int32');

% 資料類型：'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
%           'int64', 'uint64', 'single', 'double', 'char'
```

### 檔案位置

```matlab
% 取得位置
pos = ftell(fid);

% 設定位置
fseek(fid, 0, 'bof');       % 檔案開頭
fseek(fid, 0, 'eof');       % 檔案結尾
fseek(fid, offset, 'cof'); % 目前位置 + 偏移量

% 倒回開頭
frewind(fid);

% 檢查檔案結尾
tf = feof(fid);
```

### 檔案與目錄操作

```matlab
% 檢查存在
tf = exist('file.txt', 'file');
tf = exist('folder', 'dir');
tf = isfile('file.txt');
tf = isfolder('folder');

% 列出檔案
files = dir('*.csv');           % 結構體陣列
files = dir('folder/*.mat');
names = {files.name};

% 檔案資訊
info = dir('file.txt');
info.name
info.bytes
info.date
info.datenum

% 檔案操作
copyfile('src.txt', 'dst.txt');
movefile('src.txt', 'dst.txt');
delete('file.txt');

% 目錄操作
mkdir('newfolder');
rmdir('folder');
rmdir('folder', 's');           % 移除含內容
cd('path');
pwd                             % 目前目錄
```

### 路徑操作

```matlab
% 建構路徑
fullpath = fullfile('folder', 'subfolder', 'file.txt');
fullpath = fullfile(pwd, 'file.txt');

% 解析路徑
[path, name, ext] = fileparts('/path/to/file.txt');
% path = '/path/to', name = 'file', ext = '.txt'

% 暫存檔案/資料夾
tmpfile = tempname;
tmpdir = tempdir;
```
