# 矩陣與陣列參考

## 目錄
1. [陣列建立](#陣列建立)
2. [索引與下標](#索引與下標)
3. [陣列操作](#陣列操作)
4. [串接與重塑](#串接與重塑)
5. [陣列資訊](#陣列資訊)
6. [排序與搜尋](#排序與搜尋)

## 陣列建立

### 基本建立

```matlab
% 直接指定
A = [1 2 3; 4 5 6; 7 8 9];    % 3x3 矩陣（列以 ; 分隔）
v = [1, 2, 3, 4, 5];           % 列向量
v = [1; 2; 3; 4; 5];           % 欄向量

% 範圍運算子
v = 1:10;                       % 1 到 10，步長 1
v = 0:0.5:5;                    % 0 到 5，步長 0.5
v = 10:-1:1;                    % 10 倒數到 1

% 線性/對數間隔
v = linspace(0, 1, 100);        % 0 到 1 的 100 個點
v = logspace(0, 3, 50);         % 10^0 到 10^3 的 50 個點
```

### 特殊矩陣

```matlab
% 常見模式
I = eye(n);                     % n×n 單位矩陣
I = eye(m, n);                  % m×n 單位矩陣
Z = zeros(m, n);                % m×n 零矩陣
O = ones(m, n);                 % m×n 全一矩陣
D = diag([1 2 3]);              % 從向量建立對角矩陣
d = diag(A);                    % 從矩陣擷取對角線

% 隨機矩陣
R = rand(m, n);                 % 均勻分佈 [0,1]
R = randn(m, n);                % 常態分佈（均值=0，標準差=1）
R = randi([a b], m, n);         % [a,b] 中的隨機整數
R = randperm(n);                % 1:n 的隨機排列

% 邏輯陣列
T = true(m, n);                 % 全 true
F = false(m, n);                % 全 false

% 2D/3D 網格
[X, Y] = meshgrid(x, y);        % 從向量建立 2D 網格
[X, Y, Z] = meshgrid(x, y, z);  % 3D 網格
[X, Y] = ndgrid(x, y);          % 替代方案（不同方向）
```

### 從現有陣列建立

```matlab
A_like = zeros(size(B));        % 與 B 相同大小
A_like = ones(size(B), 'like', B);  % 與 B 相同大小和類型
A_copy = A;                     % 複製（值傳遞，非參考）
```

## 索引與下標

### 基本索引

```matlab
% 單一元素（從 1 開始索引）
elem = A(2, 3);                 % 第 2 列，第 3 欄
elem = A(5);                    % 線性索引（欄優先順序）

% 範圍
row = A(2, :);                  % 整個第 2 列
col = A(:, 3);                  % 整個第 3 欄
sub = A(1:2, 2:3);              % 第 1-2 列，第 2-3 欄

% end 關鍵字
last = A(end, :);               % 最後一列
last3 = A(end-2:end, :);        % 最後 3 列
```

### 邏輯索引

```matlab
% 找出符合條件的元素
idx = A > 5;                    % 邏輯陣列
elements = A(A > 5);            % 擷取 > 5 的元素
A(A < 0) = 0;                   % 將負元素設為 0

% 組合條件
idx = (A > 0) & (A < 10);       % AND
idx = (A < 0) | (A > 10);       % OR
idx = ~(A == 0);                % NOT
```

### 線性索引

```matlab
% 線性與下標索引轉換
[row, col] = ind2sub(size(A), linearIdx);
linearIdx = sub2ind(size(A), row, col);

% 找出非零/符合條件的索引
idx = find(A > 5);              % A > 5 的線性索引
idx = find(A > 5, k);           % 前 k 個索引
idx = find(A > 5, k, 'last');   % 最後 k 個索引
[row, col] = find(A > 5);       % 下標索引
```

### 進階索引

```matlab
% 使用陣列索引
rows = [1 3 5];
cols = [2 4];
sub = A(rows, cols);            % 子矩陣

% 使用另一個陣列進行邏輯索引
B = A(logical_mask);            % mask 為 true 的元素

% 帶索引的賦值
A(1:2, 1:2) = [10 20; 30 40];   % 賦值子矩陣
A(:) = 1:numel(A);              % 賦值所有元素（欄優先）
```

## 陣列操作

### 逐元素運算

```matlab
% 算術（逐元素使用 . 前綴）
C = A + B;                      % 加法
C = A - B;                      % 減法
C = A .* B;                     % 逐元素乘法
C = A ./ B;                     % 逐元素除法
C = A .\ B;                     % 逐元素左除法（B./A）
C = A .^ n;                     % 逐元素乘冪

% 比較（逐元素）
C = A == B;                     % 相等
C = A ~= B;                     % 不相等
C = A < B;                      % 小於
C = A <= B;                     % 小於等於
C = A > B;                      % 大於
C = A >= B;                     % 大於等於
```

### 矩陣運算

```matlab
% 矩陣算術
C = A * B;                      % 矩陣乘法
C = A ^ n;                      % 矩陣乘冪
C = A';                         % 共軛轉置
C = A.';                        % 轉置（無共軛）

% 矩陣函數
B = inv(A);                     % 反矩陣
B = pinv(A);                    % 偽逆
d = det(A);                     % 行列式
t = trace(A);                   % 跡（對角線總和）
r = rank(A);                    % 秩
n = norm(A);                    % 矩陣/向量範數
n = norm(A, 'fro');             % Frobenius 範數

% 求解線性系統
x = A \ b;                      % 求解 Ax = b
x = b' / A';                    % 求解 xA = b
```

### 常用函數

```matlab
% 對每個元素應用
B = abs(A);                     % 絕對值
B = sqrt(A);                    % 平方根
B = exp(A);                     % 指數
B = log(A);                     % 自然對數
B = log10(A);                   % 以 10 為底對數
B = sin(A);                     % 正弦（弧度）
B = sind(A);                    % 正弦（角度）
B = round(A);                   % 四捨五入到最近整數
B = floor(A);                   % 向下捨入
B = ceil(A);                    % 向上捨入
B = real(A);                    % 實部
B = imag(A);                    % 虛部
B = conj(A);                    % 共軛複數
```

## 串接與重塑

### 串接

```matlab
% 水平（並排）
C = [A B];                      % 串接欄
C = [A, B];                     % 同上
C = horzcat(A, B);              % 函數形式
C = cat(2, A, B);               % 沿維度 2 串接

% 垂直（堆疊）
C = [A; B];                     % 串接列
C = vertcat(A, B);              % 函數形式
C = cat(1, A, B);               % 沿維度 1 串接

% 區塊對角
C = blkdiag(A, B, C);           % 區塊對角矩陣
```

### 重塑

```matlab
% 重塑
B = reshape(A, m, n);           % 重塑為 m×n（相同總元素）
B = reshape(A, [], n);          % 自動計算列數
v = A(:);                       % 展平為欄向量

% 轉置與排列
B = A';                         % 轉置 2D
B = permute(A, [2 1 3]);        % 排列維度
B = ipermute(A, [2 1 3]);       % 逆排列

% 移除/新增維度
B = squeeze(A);                 % 移除單一維度
B = shiftdim(A, n);             % 移位維度

% 複製
B = repmat(A, m, n);            % 拼貼 m×n 次
B = repelem(A, m, n);           % 複製元素
```

### 翻轉與旋轉

```matlab
B = flip(A);                    % 沿第一個非單一維度翻轉
B = flip(A, dim);               % 沿維度 dim 翻轉
B = fliplr(A);                  % 左右翻轉（欄）
B = flipud(A);                  % 上下翻轉（列）
B = rot90(A);                   % 逆時針旋轉 90°
B = rot90(A, k);                % 旋轉 k×90°
B = circshift(A, k);            % 循環移位
```

## 陣列資訊

### 大小與維度

```matlab
[m, n] = size(A);               % 列數和欄數
m = size(A, 1);                 % 列數
n = size(A, 2);                 % 欄數
sz = size(A);                   % 大小向量
len = length(A);                % 最大維度
num = numel(A);                 % 總元素數
ndim = ndims(A);                % 維度數
```

### 類型檢查

```matlab
tf = isempty(A);                % 是否為空？
tf = isscalar(A);               % 是否為純量（1×1）？
tf = isvector(A);               % 是否為向量（1×n 或 n×1）？
tf = isrow(A);                  % 是否為列向量？
tf = iscolumn(A);               % 是否為欄向量？
tf = ismatrix(A);               % 是否為 2D 矩陣？
tf = isnumeric(A);              % 是否為數值？
tf = isreal(A);                 % 是否為實數（無虛部）？
tf = islogical(A);              % 是否為邏輯值？
tf = isnan(A);                  % 哪些元素是 NaN？
tf = isinf(A);                  % 哪些元素是 Inf？
tf = isfinite(A);               % 哪些元素是有限的？
```

### 比較

```matlab
tf = isequal(A, B);             % 陣列是否相等？
tf = isequaln(A, B);            % 相等，NaN 視為相等？
tf = all(A);                    % 全部非零/true？
tf = any(A);                    % 任何非零/true？
tf = all(A, dim);               % 沿維度全部
tf = any(A, dim);               % 沿維度任何
```

## 排序與搜尋

### 排序

```matlab
B = sort(A);                    % 欄升序排序
B = sort(A, 'descend');         % 降序排序
B = sort(A, dim);               % 沿維度排序
[B, idx] = sort(A);             % 同時返回原始索引
B = sortrows(A);                % 按第一欄排序列
B = sortrows(A, col);           % 按特定欄排序
B = sortrows(A, col, 'descend');
```

### 唯一與集合運算

```matlab
B = unique(A);                  % 唯一元素
[B, ia, ic] = unique(A);        % 帶索引資訊
B = unique(A, 'rows');          % 唯一列

% 集合運算
C = union(A, B);                % 聯集
C = intersect(A, B);            % 交集
C = setdiff(A, B);              % A - B（在 A 但不在 B）
C = setxor(A, B);               % 對稱差
tf = ismember(A, B);            % A 的每個元素是否在 B 中？
```

### 最小/最大

```matlab
m = min(A);                     % 欄最小值
m = min(A, [], 'all');          % 全域最小值
[m, idx] = min(A);              % 帶索引
m = min(A, B);                  % 逐元素最小值

M = max(A);                     % 欄最大值
M = max(A, [], 'all');          % 全域最大值
[M, idx] = max(A);              % 帶索引

[minVal, minIdx] = min(A(:));   % 全域最小值帶線性索引
[maxVal, maxIdx] = max(A(:));   % 全域最大值帶線性索引

% k 個最小/最大
B = mink(A, k);                 % k 個最小元素
B = maxk(A, k);                 % k 個最大元素
```

### 總和與乘積

```matlab
s = sum(A);                     % 欄總和
s = sum(A, 'all');              % 總計
s = sum(A, dim);                % 沿維度總和
s = cumsum(A);                  % 累積總和

p = prod(A);                    % 欄乘積
p = prod(A, 'all');             % 總乘積
p = cumprod(A);                 % 累積乘積
```
