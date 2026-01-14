# 程式設計參考

## 目錄
1. [腳本與函數](#腳本與函數)
2. [控制流程](#控制流程)
3. [函數類型](#函數類型)
4. [錯誤處理](#錯誤處理)
5. [效能與除錯](#效能與除錯)
6. [物件導向程式設計](#物件導向程式設計)

## 腳本與函數

### 腳本

```matlab
% 腳本是包含 MATLAB 命令的 .m 檔案
% 它們在基礎工作區中執行（共享變數）

% 範例：myscript.m
% 這是一個註解
x = 1:10;
y = x.^2;
plot(x, y);
title('我的繪圖');

% 執行腳本
myscript;           % 或：run('myscript.m')
```

### 函數

```matlab
% 函數有自己的工作區
% 儲存為與函數同名的檔案

% 範例：myfunction.m
function y = myfunction(x)
%MYFUNCTION 函數的簡短描述
%   Y = MYFUNCTION(X) 詳細描述
%
%   範例：
%       y = myfunction(5);
%
%   另請參閱 OTHERFUNCTION
    y = x.^2;
end

% 多輸出
function [result1, result2] = multioutput(x)
    result1 = x.^2;
    result2 = x.^3;
end

% 可變參數
function varargout = flexfun(varargin)
    % varargin 是輸入的儲存格陣列
    % varargout 是輸出的儲存格陣列
    n = nargin;          % 輸入數量
    m = nargout;         % 輸出數量
end
```

### 輸入驗證

```matlab
function result = validatedinput(x, options)
    arguments
        x (1,:) double {mustBePositive}
        options.Normalize (1,1) logical = false
        options.Scale (1,1) double {mustBePositive} = 1
    end

    result = x * options.Scale;
    if options.Normalize
        result = result / max(result);
    end
end

% 使用方式
y = validatedinput([1 2 3], 'Normalize', true, 'Scale', 2);

% 常用驗證器
% mustBePositive, mustBeNegative, mustBeNonzero
% mustBeInteger, mustBeNumeric, mustBeFinite
% mustBeNonNaN, mustBeReal, mustBeNonempty
% mustBeMember, mustBeInRange, mustBeGreaterThan
```

### 區域函數

```matlab
% 區域函數出現在主函數之後
% 只能在同一檔案內存取

function result = mainfunction(x)
    intermediate = helper1(x);
    result = helper2(intermediate);
end

function y = helper1(x)
    y = x.^2;
end

function y = helper2(x)
    y = sqrt(x);
end
```

## 控制流程

### 條件語句

```matlab
% if-elseif-else
if condition1
    % 語句
elseif condition2
    % 語句
else
    % 語句
end

% 邏輯運算子
%   &  - AND（逐元素）
%   |  - OR（逐元素）
%   ~  - NOT
%   && - AND（短路，純量）
%   || - OR（短路，純量）
%   == - 相等
%   ~= - 不相等
%   <, <=, >, >= - 比較

% 範例
if x > 0 && y > 0
    quadrant = 1;
elseif x < 0 && y > 0
    quadrant = 2;
elseif x < 0 && y < 0
    quadrant = 3;
else
    quadrant = 4;
end
```

### Switch 語句

```matlab
switch expression
    case value1
        % 語句
    case {value2, value3}  % 多個值
        % 語句
    otherwise
        % 預設語句
end

% 範例
switch dayOfWeek
    case {'Saturday', 'Sunday'}
        dayType = '週末';
    case {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
        dayType = '工作日';
    otherwise
        dayType = '未知';
end
```

### For 迴圈

```matlab
% 基本 for 迴圈
for i = 1:10
    % 使用 i 的語句
end

% 自訂步長
for i = 10:-1:1
    % 倒數
end

% 遍歷向量
for val = [1 3 5 7 9]
    % val 取每個值
end

% 遍歷矩陣的欄
for col = A
    % col 是欄向量
end

% 遍歷儲存格陣列
for i = 1:length(C)
    item = C{i};
end
```

### While 迴圈

```matlab
% 基本 while 迴圈
while condition
    % 語句
    % 更新條件
end

% 範例
count = 0;
while count < 10
    count = count + 1;
    % 做某事
end
```

### 迴圈控制

```matlab
% Break - 立即退出迴圈
for i = 1:100
    if someCondition
        break;
    end
end

% Continue - 跳到下一次迭代
for i = 1:100
    if skipCondition
        continue;
    end
    % 處理 i
end

% Return - 退出函數
function y = myfunction(x)
    if x < 0
        y = NaN;
        return;
    end
    y = sqrt(x);
end
```

## 函數類型

### 匿名函數

```matlab
% 建立內嵌函數
f = @(x) x.^2 + 2*x + 1;
g = @(x, y) x.^2 + y.^2;

% 使用
y = f(5);           % 36
z = g(3, 4);        % 25

% 帶捕獲變數
a = 2;
h = @(x) a * x;     % 捕獲 a 的當前值
y = h(5);           % 10
a = 3;              % 改變 a 不影響 h
y = h(5);           % 仍然是 10

% 無參數
now_fn = @() datestr(now);
timestamp = now_fn();

% 傳遞給其他函數
result = integral(f, 0, 1);
```

### 巢狀函數

```matlab
function result = outerfunction(x)
    y = x.^2;           % 與巢狀函數共享

    function z = nestedfunction(a)
        z = y + a;      % 可存取外部作用域的 y
    end

    result = nestedfunction(10);
end
```

### 函數控制代碼

```matlab
% 建立現有函數的控制代碼
h = @sin;
y = h(pi/2);        % 1

% 從字串
h = str2func('cos');

% 取得函數名稱
name = func2str(h);

% 取得區域函數的控制代碼
handles = localfunctions;

% 函數資訊
info = functions(h);
```

### 回呼

```matlab
% 使用函數控制代碼作為回呼

% 計時器範例
t = timer('TimerFcn', @myCallback, 'Period', 1);
start(t);

function myCallback(~, ~)
    disp(['時間：' datestr(now)]);
end

% 使用匿名函數
t = timer('TimerFcn', @(~,~) disp('滴答'), 'Period', 1);

% GUI 回呼
uicontrol('Style', 'pushbutton', 'Callback', @buttonPressed);
```

## 錯誤處理

### Try-Catch

```matlab
try
    % 可能出錯的程式碼
    result = riskyOperation();
catch ME
    % 處理錯誤
    disp(['錯誤：' ME.message]);
    disp(['識別碼：' ME.identifier]);

    % 可選重新拋出
    rethrow(ME);
end

% 捕獲特定錯誤
try
    result = operation();
catch ME
    switch ME.identifier
        case 'MATLAB:divideByZero'
            result = Inf;
        case 'MATLAB:nomem'
            rethrow(ME);
        otherwise
            result = NaN;
    end
end
```

### 拋出錯誤

```matlab
% 簡單錯誤
error('發生錯誤');

% 帶識別碼
error('MyPkg:InvalidInput', '輸入必須為正');

% 帶格式化
error('MyPkg:OutOfRange', '值 %f 超出範圍 [%f, %f]', val, lo, hi);

% 建立並拋出例外
ME = MException('MyPkg:Error', '錯誤訊息');
throw(ME);

% 斷言
assert(condition, '若為 false 的訊息');
assert(x > 0, 'MyPkg:NotPositive', 'x 必須為正');
```

### 警告

```matlab
% 發出警告
warning('這可能是個問題');
warning('MyPkg:Warning', '警告訊息');

% 控制警告
warning('off', 'MyPkg:Warning');    % 停用特定警告
warning('on', 'MyPkg:Warning');     % 啟用
warning('off', 'all');              % 停用所有
warning('on', 'all');               % 啟用所有

% 查詢警告狀態
s = warning('query', 'MyPkg:Warning');

% 暫時停用
origState = warning('off', 'MATLAB:nearlySingularMatrix');
% ... 程式碼 ...
warning(origState);
```

## 效能與除錯

### 計時

```matlab
% 簡單計時
tic;
% ... 程式碼 ...
elapsed = toc;

% 多個計時器
t1 = tic;
% ... 程式碼 ...
elapsed1 = toc(t1);

% CPU 時間
t = cputime;
% ... 程式碼 ...
cpuElapsed = cputime - t;

% 分析器
profile on;
myfunction();
profile viewer;     % 分析結果的 GUI
p = profile('info'); % 取得程式化結果
profile off;
```

### 記憶體

```matlab
% 記憶體資訊
[user, sys] = memory;   % 僅 Windows
whos;                   % 變數大小

% 清除變數
clear x y z;
clear all;              % 所有變數（謹慎使用）
clearvars -except x y;  % 保留特定變數
```

### 除錯

```matlab
% 設定中斷點（在編輯器或程式化）
dbstop in myfunction at 10
dbstop if error
dbstop if warning
dbstop if naninf          % 在 NaN 或 Inf 時停止

% 逐步執行程式碼
dbstep                    % 下一行
dbstep in                 % 進入函數
dbstep out                % 離開函數
dbcont                    % 繼續執行
dbquit                    % 退出除錯

% 清除中斷點
dbclear all

% 檢查狀態
dbstack                   % 呼叫堆疊
whos                      % 變數
```

### 向量化技巧

```matlab
% 盡可能避免迴圈
% 慢：
for i = 1:n
    y(i) = x(i)^2;
end

% 快：
y = x.^2;

% 逐元素運算（使用 . 前綴）
y = a .* b;             % 逐元素乘法
y = a ./ b;             % 逐元素除法
y = a .^ b;             % 逐元素乘冪

% 內建函數對陣列運算
y = sin(x);             % 對所有元素應用
s = sum(x);             % 全部加總
m = max(x);             % 最大值

% 邏輯索引代替 find
% 慢：
idx = find(x > 0);
y = x(idx);

% 快：
y = x(x > 0);

% 預分配陣列
% 慢：
y = [];
for i = 1:n
    y(i) = compute(i);
end

% 快：
y = zeros(1, n);
for i = 1:n
    y(i) = compute(i);
end
```

### 平行運算

```matlab
% 平行 for 迴圈
parfor i = 1:n
    results(i) = compute(i);
end

% 注意：parfor 有限制
% - 迭代必須獨立
% - 變數分類（切片、廣播等）

% 啟動平行池
pool = parpool;         % 預設叢集
pool = parpool(4);      % 4 個工作者

% 刪除池
delete(gcp('nocreate'));

% 平行陣列運算
spmd
    % 每個工作者執行此區塊
    localData = myData(labindex);
    result = process(localData);
end
```

## 物件導向程式設計

### 類別定義

```matlab
% 在 MyClass.m 檔案中
classdef MyClass
    properties
        PublicProp
    end

    properties (Access = private)
        PrivateProp
    end

    properties (Constant)
        ConstProp = 42
    end

    methods
        % 建構函數
        function obj = MyClass(value)
            obj.PublicProp = value;
        end

        % 實例方法
        function result = compute(obj, x)
            result = obj.PublicProp * x;
        end
    end

    methods (Static)
        function result = staticMethod(x)
            result = x.^2;
        end
    end
end
```

### 使用類別

```matlab
% 建立物件
obj = MyClass(10);

% 存取屬性
val = obj.PublicProp;
obj.PublicProp = 20;

% 呼叫方法
result = obj.compute(5);
result = compute(obj, 5);   % 等效

% 靜態方法
result = MyClass.staticMethod(3);

% 常數屬性
val = MyClass.ConstProp;
```

### 繼承

```matlab
classdef DerivedClass < BaseClass
    properties
        ExtraProp
    end

    methods
        function obj = DerivedClass(baseVal, extraVal)
            % 呼叫父類別建構函數
            obj@BaseClass(baseVal);
            obj.ExtraProp = extraVal;
        end

        % 覆寫方法
        function result = compute(obj, x)
            % 呼叫父類別方法
            baseResult = compute@BaseClass(obj, x);
            result = baseResult + obj.ExtraProp;
        end
    end
end
```

### Handle 與 Value 類別

```matlab
% Value 類別（預設）- 複製語意
classdef ValueClass
    properties
        Data
    end
end

a = ValueClass();
a.Data = 1;
b = a;          % b 是副本
b.Data = 2;     % a.Data 仍然是 1

% Handle 類別 - 參考語意
classdef HandleClass < handle
    properties
        Data
    end
end

a = HandleClass();
a.Data = 1;
b = a;          % b 參考同一物件
b.Data = 2;     % a.Data 現在是 2
```

### 事件與監聽器

```matlab
classdef EventClass < handle
    events
        DataChanged
    end

    properties
        Data
    end

    methods
        function set.Data(obj, value)
            obj.Data = value;
            notify(obj, 'DataChanged');
        end
    end
end

% 使用方式
obj = EventClass();
listener = addlistener(obj, 'DataChanged', @(src, evt) disp('資料已變更！'));
obj.Data = 42;  % 觸發事件
```
