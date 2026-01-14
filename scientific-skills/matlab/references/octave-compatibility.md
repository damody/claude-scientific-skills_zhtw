# GNU Octave 相容性參考

## 目錄
1. [概述](#概述)
2. [語法差異](#語法差異)
3. [運算子差異](#運算子差異)
4. [函數差異](#函數差異)
5. [Octave 獨有功能](#octave-獨有功能)
6. [Octave 缺少的功能](#octave-缺少的功能)
7. [撰寫相容程式碼](#撰寫相容程式碼)
8. [Octave 套件](#octave-套件)

## 概述

GNU Octave 是 MATLAB 的免費、開源替代品，具有高度相容性。大多數 MATLAB 腳本可以在 Octave 中執行，無需或只需極少修改。但是，仍有一些需要注意的差異。

### 安裝

```bash
# macOS (Homebrew)
brew install octave

# Ubuntu/Debian
sudo apt install octave

# Fedora
sudo dnf install octave

# Windows
# 從 https://octave.org/download 下載安裝程式
```

### 執行 Octave

```bash
# 互動模式
octave

# 執行腳本
octave script.m
octave --eval "disp('Hello')"

# GUI 模式
octave --gui

# 僅命令列（無圖形）
octave --no-gui
octave-cli
```

## 語法差異

### 註解

```matlab
% MATLAB 風格（兩者皆可）
% 這是一個註解

# Octave 風格（僅 Octave）
# 這在 Octave 中也是註解

% 為了相容性，請始終使用 %
```

### 字串引號

```matlab
% MATLAB：僅單引號（字元陣列）
str = 'Hello';              % 字元陣列
str = "Hello";              % 字串（R2017a+）

% Octave：兩者皆可，但行為不同
str1 = 'Hello';             % 字元陣列，無跳脫序列
str2 = "Hello\n";           % 將 \n 解譯為換行

% 為了相容性，字元陣列請使用單引號
% 避免帶跳脫序列的雙引號
```

### 行續接

```matlab
% MATLAB 風格（兩者皆可）
x = 1 + 2 + 3 + ...
    4 + 5;

% Octave 也接受反斜線
x = 1 + 2 + 3 + \
    4 + 5;

% 為了相容性，請使用 ...
```

### 區塊終止符

```matlab
% MATLAB 風格（兩者皆可）
if condition
    % 程式碼
end

for i = 1:10
    % 程式碼
end

% Octave 也接受特定終止符
if condition
    # 程式碼
endif

for i = 1:10
    # 程式碼
endfor

while condition
    # 程式碼
endwhile

% 為了相容性，請始終使用 'end'
```

### 函數定義

```matlab
% MATLAB 要求函數放在同名檔案中
% Octave 允許命令列函數定義

% Octave 命令列函數
function y = f(x)
    y = x^2;
endfunction

% 為了相容性，請在 .m 檔案中定義函數
```

## 運算子差異

### 遞增/遞減運算子

```matlab
% Octave 有 C 風格運算子（MATLAB 沒有）
x++;                        % x = x + 1
x--;                        % x = x - 1
++x;                        % 前置遞增
--x;                        % 前置遞減

% 為了相容性，請使用明確賦值
x = x + 1;
x = x - 1;
```

### 複合賦值

```matlab
% Octave 支援（MATLAB 不支援）
x += 5;                     % x = x + 5
x -= 3;                     % x = x - 3
x *= 2;                     % x = x * 2
x /= 4;                     % x = x / 4
x ^= 2;                     % x = x ^ 2

% 逐元素版本
x .+= y;
x .-= y;
x .*= y;
x ./= y;
x .^= y;

% 為了相容性，請使用明確賦值
x = x + 5;
x = x .* y;
```

### 邏輯運算子

```matlab
% 兩者皆支援
& | ~ && ||

% 短路行為差異：
% MATLAB：& 和 | 在 if/while 條件中會短路
% Octave：僅 && 和 || 短路

% 為了可預測行為，請使用：
% && || 用於純量短路邏輯
% & | 用於逐元素運算
```

### 運算式後立即索引

```matlab
% Octave 允許運算式後立即索引
result = sin(x)(1:10);      % sin(x) 的前 10 個元素
value = func(arg).field;    % 存取回傳結構體的欄位

% MATLAB 需要中間變數
temp = sin(x);
result = temp(1:10);

temp = func(arg);
value = temp.field;

% 為了相容性，請使用中間變數
```

## 函數差異

### 內建函數

大多數基本函數是相容的。一些差異：

```matlab
% 函數名稱差異
% MATLAB          Octave 替代
% ------          ------------------
% inputname       （不可用）
% inputParser     （部分支援）
% validateattributes  （部分支援）

% 邊界情況的行為差異
% 請查閱特定函數的文件
```

### 隨機數生成

```matlab
% 兩者預設使用 Mersenne Twister
% 種子設定類似
rng(42);                    % MATLAB
rand('seed', 42);           % Octave（也接受 rng 語法）

% 為了相容性
rng(42);                    % 在現代 Octave 中可用
```

### 圖形

```matlab
% 基本繪圖是相容的
plot(x, y);
xlabel('X'); ylabel('Y');
title('標題');
legend('資料');

% 一些進階功能不同
% - Octave 使用 gnuplot 或 Qt 圖形
% - 某些屬性名稱可能不同
% - 動畫/GUI 功能有所差異

% 在兩個環境中測試圖形程式碼
```

### 檔案 I/O

```matlab
% 基本 I/O 是相容的
save('file.mat', 'x', 'y');
load('file.mat');
dlmread('file.txt');
dlmwrite('file.txt', data);

% MAT 檔案版本
save('file.mat', '-v7');    % 相容格式
save('file.mat', '-v7.3');  % HDF5 格式（Octave 部分支援）

% 為了相容性，請使用 -v7 或 -v6
```

## Octave 獨有功能

### do-until 迴圈

```matlab
% 僅 Octave
do
    x = x + 1;
until (x > 10)

% 等效的 MATLAB/相容程式碼
x = x + 1;
while x <= 10
    x = x + 1;
end
```

### unwind_protect

```matlab
% 僅 Octave - 保證清理
unwind_protect
    % 可能出錯的程式碼
    result = risky_operation();
unwind_protect_cleanup
    % 始終執行（類似 finally）
    cleanup();
end_unwind_protect

% MATLAB 等效
try
    result = risky_operation();
catch
end
cleanup();  % 若錯誤未捕獲則不保證
```

### 內建文件

```matlab
% Octave 支援函數中的 Texinfo 文件
function y = myfunction(x)
    %% -*- texinfo -*-
    %% @deftypefn {Function File} {@var{y} =} myfunction (@var{x})
    %% myfunction 的描述。
    %% @end deftypefn
    y = x.^2;
endfunction
```

### 套件系統

```matlab
% Octave Forge 套件
pkg install -forge control
pkg load control

% 列出已安裝套件
pkg list

% 為了 MATLAB 相容性，請使用等效工具箱
% 或直接包含套件功能
```

## Octave 缺少的功能

### Simulink

```matlab
% 無 Octave 等效物
% Simulink 模型（.slx、.mdl）無法在 Octave 中執行
```

### MATLAB 工具箱

```matlab
% 許多工具箱函數不可用
% 某些有 Octave Forge 等效物：

% MATLAB 工具箱        Octave Forge 套件
% ---------------       --------------------
% Control System        control
% Signal Processing     signal
% Image Processing      image
% Statistics            statistics
% Optimization          optim

% 使用 pkg list 查看可用套件
```

### App Designer / GUIDE

```matlab
% MATLAB GUI 工具在 Octave 中不可用
% Octave 有基本 UI 函數：
uicontrol, uimenu, figure 屬性

% 對於跨平台 GUI，請考慮：
% - 基於 Web 的介面
% - Qt（透過 Octave 的 Qt 圖形）
```

### 物件導向程式設計

```matlab
% Octave 有部分 classdef 支援
% 某些功能缺少或行為不同：
% - Handle 類別事件
% - 屬性驗證
% - 某些存取修飾符

% 為了相容性，請使用較簡單的 OOP 模式
% 或基於結構體的方法
```

### Live 腳本

```matlab
% .mlx 檔案僅適用於 MATLAB
% 為了相容性請使用一般 .m 腳本
```

## 撰寫相容程式碼

### 偵測

```matlab
function tf = isOctave()
    tf = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end

% 用於條件程式碼
if isOctave()
    % Octave 特定程式碼
else
    % MATLAB 特定程式碼
end
```

### 最佳實踐

```matlab
% 1. 註解使用 %，不要用 #
% 良好
% 這是一個註解

% 避免
# 這是一個註解（僅 Octave）

% 2. 行續接使用 ...
% 良好
x = 1 + 2 + 3 + ...
    4 + 5;

% 避免
x = 1 + 2 + 3 + \
    4 + 5;

% 3. 所有區塊使用 'end'
% 良好
if condition
    code
end

% 避免
if condition
    code
endif

% 4. 避免複合運算子
% 良好
x = x + 1;

% 避免
x++;
x += 1;

% 5. 字串使用單引號
% 良好
str = 'Hello World';

% 避免（跳脫序列問題）
str = "Hello\nWorld";

% 6. 索引使用中間變數
% 良好
temp = func(arg);
result = temp(1:10);

% 避免（僅 Octave）
result = func(arg)(1:10);

% 7. MAT 檔案使用相容格式儲存
save('data.mat', 'x', 'y', '-v7');
```

### 測試相容性

```bash
# 在兩個環境中測試
matlab -nodisplay -nosplash -r "run('test_script.m'); exit;"
octave --no-gui test_script.m

# 建立測試腳本
# test_script.m:
# try
#     main_function();
#     disp('測試通過');
# catch ME
#     disp(['測試失敗：' ME.message]);
# end
```

## Octave 套件

### 安裝套件

```matlab
% 從 Octave Forge 安裝
pkg install -forge package_name

% 從檔案安裝
pkg install package_file.tar.gz

% 從 URL 安裝
pkg install 'http://example.com/package.tar.gz'

% 解除安裝
pkg uninstall package_name
```

### 使用套件

```matlab
% 載入套件（使用前必須）
pkg load control
pkg load signal
pkg load image

% 啟動時載入（加入 .octaverc）
pkg load control

% 列出已載入套件
pkg list

% 卸載套件
pkg unload control
```

### 常用套件

| 套件 | 描述 |
|---------|-------------|
| control | 控制系統設計 |
| signal | 訊號處理 |
| image | 影像處理 |
| statistics | 統計函數 |
| optim | 最佳化演算法 |
| io | 輸入/輸出函數 |
| struct | 結構體操作 |
| symbolic | 符號數學（透過 SymPy） |
| parallel | 平行運算 |
| netcdf | NetCDF 檔案支援 |

### 套件管理

```matlab
% 更新所有套件
pkg update

% 取得套件描述
pkg describe package_name

% 檢查更新
pkg list  % 與 Octave Forge 網站比較
```
