# 數學參考

## 目錄
1. [線性代數](#線性代數)
2. [基本數學](#基本數學)
3. [微積分與積分](#微積分與積分)
4. [微分方程](#微分方程)
5. [最佳化](#最佳化)
6. [統計學](#統計學)
7. [訊號處理](#訊號處理)
8. [內插與擬合](#內插與擬合)

## 線性代數

### 求解線性系統

```matlab
% Ax = b
x = A \ b;                      % 建議方法（mldivide）
x = linsolve(A, b);             % 帶選項
x = inv(A) * b;                 % 效率較低，避免使用

% linsolve 的選項
opts.LT = true;                 % 下三角
opts.UT = true;                 % 上三角
opts.SYM = true;                % 對稱
opts.POSDEF = true;             % 正定
x = linsolve(A, b, opts);

% xA = b
x = b / A;                      % mrdivide

% 最小平方（超定系統）
x = A \ b;                      % 最小範數解
x = lsqminnorm(A, b);           % 明確最小範數

% 非負最小平方
x = lsqnonneg(A, b);            % x >= 0 約束
```

### 矩陣分解

```matlab
% LU 分解：A = L*U 或 P*A = L*U
[L, U] = lu(A);                 % L 可能不是下三角
[L, U, P] = lu(A);              % P*A = L*U

% QR 分解：A = Q*R
[Q, R] = qr(A);                 % 完整分解
[Q, R] = qr(A, 0);              % 經濟大小
[Q, R, P] = qr(A);              % 欄樞軸：A*P = Q*R

% Cholesky：A = R'*R（對稱正定）
R = chol(A);                    % 上三角
L = chol(A, 'lower');           % 下三角

% LDL'：A = L*D*L'（對稱）
[L, D] = ldl(A);

% Schur 分解：A = U*T*U'
[U, T] = schur(A);              % T 是準三角
[U, T] = schur(A, 'complex');   % T 是三角
```

### 特徵值與特徵向量

```matlab
% 特徵值
e = eig(A);                     % 僅特徵值
[V, D] = eig(A);                % V：特徵向量，D：對角特徵值
                                % A*V = V*D

% 廣義特徵值：A*v = lambda*B*v
e = eig(A, B);
[V, D] = eig(A, B);

% 稀疏/大矩陣（特徵值子集）
e = eigs(A, k);                 % k 個最大模
e = eigs(A, k, 'smallestabs');  % k 個最小模
[V, D] = eigs(A, k, 'largestreal');
```

### 奇異值分解

```matlab
% SVD：A = U*S*V'
[U, S, V] = svd(A);             % 完整分解
[U, S, V] = svd(A, 'econ');     % 經濟大小
s = svd(A);                     % 僅奇異值

% 稀疏/大矩陣
[U, S, V] = svds(A, k);         % k 個最大奇異值

% 應用
r = rank(A);                    % 秩（計算非零奇異值）
p = pinv(A);                    % 偽逆（經由 SVD）
n = norm(A, 2);                 % 2-範數 = 最大奇異值
c = cond(A);                    % 條件數 = 最大/最小比值
```

### 矩陣性質

```matlab
d = det(A);                     % 行列式
t = trace(A);                   % 跡（對角線總和）
r = rank(A);                    % 秩
n = norm(A);                    % 2-範數（預設）
n = norm(A, 1);                 % 1-範數（最大欄總和）
n = norm(A, inf);               % 無窮範數（最大列總和）
n = norm(A, 'fro');             % Frobenius 範數
c = cond(A);                    % 條件數
c = rcond(A);                   % 倒數條件數（快速估計）
```

## 基本數學

### 三角函數

```matlab
% 弧度
y = sin(x);   y = cos(x);   y = tan(x);
y = asin(x);  y = acos(x);  y = atan(x);
y = atan2(y, x);            % 四象限反正切

% 角度
y = sind(x);  y = cosd(x);  y = tand(x);
y = asind(x); y = acosd(x); y = atand(x);

% 雙曲
y = sinh(x);  y = cosh(x);  y = tanh(x);
y = asinh(x); y = acosh(x); y = atanh(x);

% 正割、餘割、餘切
y = sec(x);   y = csc(x);   y = cot(x);
```

### 指數與對數

```matlab
y = exp(x);                     % e^x
y = log(x);                     % 自然對數（ln）
y = log10(x);                   % 以 10 為底對數
y = log2(x);                    % 以 2 為底對數
y = log1p(x);                   % log(1+x)，小 x 時精確
[F, E] = log2(x);               % F * 2^E = x

y = sqrt(x);                    % 平方根
y = nthroot(x, n);              % 實數 n 次方根
y = realsqrt(x);                % 實數平方根（若 x < 0 則錯誤）

y = pow2(x);                    % 2^x
y = x .^ y;                     % 逐元素乘冪
```

### 複數

```matlab
z = complex(a, b);              % a + bi
z = 3 + 4i;                     % 直接建立

r = real(z);                    % 實部
i = imag(z);                    % 虛部
m = abs(z);                     % 模
p = angle(z);                   % 相角（弧度）
c = conj(z);                    % 共軛複數

[theta, rho] = cart2pol(x, y);  % 直角座標轉極座標
[x, y] = pol2cart(theta, rho);  % 極座標轉直角座標
```

### 捨入與餘數

```matlab
y = round(x);                   % 四捨五入到最近整數
y = round(x, n);                % 四捨五入到 n 位小數
y = floor(x);                   % 向 -infinity 捨入
y = ceil(x);                    % 向 +infinity 捨入
y = fix(x);                     % 向零捨入

y = mod(x, m);                  % 模（m 的符號）
y = rem(x, m);                  % 餘數（x 的符號）
[q, r] = deconv(x, m);          % 商和餘數

y = sign(x);                    % 符號（-1、0 或 1）
y = abs(x);                     % 絕對值
```

### 特殊函數

```matlab
y = gamma(x);                   % Gamma 函數
y = gammaln(x);                 % 對數 gamma（避免溢位）
y = factorial(n);               % n!
y = nchoosek(n, k);             % 二項式係數

y = erf(x);                     % 誤差函數
y = erfc(x);                    % 互補誤差函數
y = erfcinv(x);                 % 反互補誤差函數

y = besselj(nu, x);             % Bessel J
y = bessely(nu, x);             % Bessel Y
y = besseli(nu, x);             % 修正 Bessel I
y = besselk(nu, x);             % 修正 Bessel K

y = legendre(n, x);             % Legendre 多項式
```

## 微積分與積分

### 數值積分

```matlab
% 定積分
q = integral(fun, a, b);        % 從 a 到 b 積分 fun
q = integral(@(x) x.^2, 0, 1);  % 範例：x^2 的積分

% 選項
q = integral(fun, a, b, 'AbsTol', 1e-10);
q = integral(fun, a, b, 'RelTol', 1e-6);

% 瑕積分
q = integral(fun, 0, Inf);      % 積分到無窮大
q = integral(fun, -Inf, Inf);   % 整個實數線

% 多維
q = integral2(fun, xa, xb, ya, yb);  % 雙重積分
q = integral3(fun, xa, xb, ya, yb, za, zb);  % 三重積分

% 從離散資料
q = trapz(x, y);                % 梯形法則
q = trapz(y);                   % 單位間距
q = cumtrapz(x, y);             % 累積積分
```

### 數值微分

```matlab
% 有限差分
dy = diff(y);                   % 一階差分
dy = diff(y, n);                % n 階差分
dy = diff(y, n, dim);           % 沿維度

% 梯度（數值導數）
g = gradient(y);                % dy/dx，單位間距
g = gradient(y, h);             % dy/dx，間距 h
[gx, gy] = gradient(Z, hx, hy); % 2D 資料的梯度
```

## 微分方程

### ODE 求解器

```matlab
% 標準形式：dy/dt = f(t, y)
odefun = @(t, y) -2*y;          % 範例：dy/dt = -2y
[t, y] = ode45(odefun, tspan, y0);

% 求解器選擇：
% ode45  - 非剛性，中等精度（預設選擇）
% ode23  - 非剛性，低精度
% ode113 - 非剛性，變階
% ode15s - 剛性，變階（若 ode45 慢則嘗試）
% ode23s - 剛性，低階
% ode23t - 中等剛性，梯形
% ode23tb - 剛性，TR-BDF2

% 帶選項
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
options = odeset('MaxStep', 0.1);
options = odeset('Events', @myEventFcn);  % 停止條件
[t, y] = ode45(odefun, tspan, y0, options);
```

### 高階 ODE

```matlab
% y'' + 2y' + y = 0, y(0) = 1, y'(0) = 0
% 轉換為系統：y1 = y, y2 = y'
% y1' = y2
% y2' = -2*y2 - y1

odefun = @(t, y) [y(2); -2*y(2) - y(1)];
y0 = [1; 0];                    % [y(0); y'(0)]
[t, y] = ode45(odefun, [0 10], y0);
plot(t, y(:,1));                % 繪製 y（第一分量）
```

### 邊界值問題

```matlab
% y'' + |y| = 0, y(0) = 0, y(4) = -2
solinit = bvpinit(linspace(0, 4, 5), [0; 0]);
sol = bvp4c(@odefun, @bcfun, solinit);

function dydx = odefun(x, y)
    dydx = [y(2); -abs(y(1))];
end

function res = bcfun(ya, yb)
    res = [ya(1); yb(1) + 2];   % y(0) = 0, y(4) = -2
end
```

## 最佳化

### 無約束最佳化

```matlab
% 單變數，有界
[x, fval] = fminbnd(fun, x1, x2);
[x, fval] = fminbnd(@(x) x.^2 - 4*x, 0, 5);

% 多變數，無約束
[x, fval] = fminsearch(fun, x0);
options = optimset('TolX', 1e-8, 'TolFun', 1e-8);
[x, fval] = fminsearch(fun, x0, options);

% 顯示迭代
options = optimset('Display', 'iter');
```

### 求根

```matlab
% 找 f(x) = 0 的位置
x = fzero(fun, x0);             % 在 x0 附近
x = fzero(fun, [x1 x2]);        % 在區間 [x1, x2]
x = fzero(@(x) cos(x) - x, 0.5);

% 多項式根
r = roots([1 0 -4]);            % x^2 - 4 = 0 的根
                                % 返回 [2; -2]
```

### 最小平方

```matlab
% 線性最小平方：最小化 ||Ax - b||
x = A \ b;                      % 標準解
x = lsqminnorm(A, b);           % 最小範數解

% 非負最小平方
x = lsqnonneg(A, b);            % x >= 0

% 非線性最小平方
x = lsqnonlin(fun, x0);         % 最小化 sum(fun(x).^2)
x = lsqcurvefit(fun, x0, xdata, ydata);  % 曲線擬合
```

## 統計學

### 描述統計

```matlab
% 集中趨勢
m = mean(x);                    % 算術平均
m = mean(x, 'all');             % 所有元素的平均
m = mean(x, dim);               % 沿維度的平均
m = mean(x, 'omitnan');         % 忽略 NaN 值
gm = geomean(x);                % 幾何平均
hm = harmmean(x);               % 調和平均
med = median(x);                % 中位數
mo = mode(x);                   % 眾數

% 離散程度
s = std(x);                     % 標準差（N-1）
s = std(x, 1);                  % 母體標準差（N）
v = var(x);                     % 變異數
r = range(x);                   % max - min
iqr_val = iqr(x);               % 四分位距

% 極值
[minv, mini] = min(x);
[maxv, maxi] = max(x);
[lo, hi] = bounds(x);           % 最小和最大一起
```

### 相關與共變異

```matlab
% 相關
R = corrcoef(X, Y);             % 相關矩陣
r = corrcoef(x, y);             % 相關係數

% 共變異
C = cov(X, Y);                  % 共變異矩陣
c = cov(x, y);                  % 共變異

% 互相關（訊號處理）
[r, lags] = xcorr(x, y);        % 互相關
[r, lags] = xcorr(x, y, 'coeff');  % 正規化
```

### 百分位數與分位數

```matlab
p = prctile(x, [25 50 75]);     % 百分位數
q = quantile(x, [0.25 0.5 0.75]);  % 分位數
```

### 移動統計

```matlab
y = movmean(x, k);              % k 點移動平均
y = movmedian(x, k);            % 移動中位數
y = movstd(x, k);               % 移動標準差
y = movvar(x, k);               % 移動變異數
y = movmin(x, k);               % 移動最小
y = movmax(x, k);               % 移動最大
y = movsum(x, k);               % 移動總和

% 視窗選項
y = movmean(x, [kb kf]);        % kb 向後，kf 向前
y = movmean(x, k, 'omitnan');   % 忽略 NaN
```

### 直方圖與分佈

```matlab
% 直方圖計數
[N, edges] = histcounts(x);     % 自動分組
[N, edges] = histcounts(x, nbins);  % 指定分組數
[N, edges] = histcounts(x, edges);  % 指定邊界

% 機率/正規化
[N, edges] = histcounts(x, 'Normalization', 'probability');
[N, edges] = histcounts(x, 'Normalization', 'pdf');

% 2D 直方圖
[N, xedges, yedges] = histcounts2(x, y);
```

## 訊號處理

### 傅立葉變換

```matlab
% FFT
Y = fft(x);                     % 1D FFT
Y = fft(x, n);                  % n 點 FFT（補零/截斷）
Y = fft2(X);                    % 2D FFT
Y = fftn(X);                    % N-D FFT

% 逆 FFT
x = ifft(Y);
X = ifft2(Y);
X = ifftn(Y);

% 將零頻率移到中心
Y_shifted = fftshift(Y);
Y = ifftshift(Y_shifted);

% 頻率軸
n = length(x);
fs = 1000;                      % 取樣頻率
f = (0:n-1) * fs / n;           % 頻率向量
f = (-n/2:n/2-1) * fs / n;      % 中心化頻率向量
```

### 濾波

```matlab
% 1D 濾波
y = filter(b, a, x);            % 應用 IIR/FIR 濾波器
y = filtfilt(b, a, x);          % 零相位濾波

% 簡單移動平均
b = ones(1, k) / k;
y = filter(b, 1, x);

% 卷積
y = conv(x, h);                 % 完整卷積
y = conv(x, h, 'same');         % 與 x 相同大小
y = conv(x, h, 'valid');        % 僅有效部分

% 反卷積
[q, r] = deconv(y, h);          % y = conv(q, h) + r

% 2D 濾波
Y = filter2(H, X);              % 2D 濾波器
Y = conv2(X, H, 'same');        % 2D 卷積
```

## 內插與擬合

### 內插

```matlab
% 1D 內插
yi = interp1(x, y, xi);         % 線性（預設）
yi = interp1(x, y, xi, 'spline');  % 樣條
yi = interp1(x, y, xi, 'pchip');   % 分段三次
yi = interp1(x, y, xi, 'nearest'); % 最近鄰

% 2D 內插
zi = interp2(X, Y, Z, xi, yi);
zi = interp2(X, Y, Z, xi, yi, 'spline');

% 3D 內插
vi = interp3(X, Y, Z, V, xi, yi, zi);

% 散佈資料
F = scatteredInterpolant(x, y, v);
vi = F(xi, yi);
```

### 多項式擬合

```matlab
% 多項式擬合
p = polyfit(x, y, n);           % 擬合 n 階多項式
                                % p = [p1, p2, ..., pn+1]
                                % y = p1*x^n + p2*x^(n-1) + ... + pn+1

% 評估多項式
yi = polyval(p, xi);

% 帶擬合品質
[p, S] = polyfit(x, y, n);
[yi, delta] = polyval(p, xi, S);  % delta = 誤差估計

% 多項式運算
r = roots(p);                   % 求根
p = poly(r);                    % 從根建立多項式
q = polyder(p);                 % 導數
q = polyint(p);                 % 積分
c = conv(p1, p2);               % 多項式相乘
[q, r] = deconv(p1, p2);        % 多項式相除
```

### 曲線擬合

```matlab
% 使用 fit 函數（曲線擬合工具箱或基本形式）
% 線性：y = a*x + b
p = polyfit(x, y, 1);
a = p(1); b = p(2);

% 指數：y = a*exp(b*x)
% 線性化：log(y) = log(a) + b*x
p = polyfit(x, log(y), 1);
b = p(1); a = exp(p(2));

% 冪次：y = a*x^b
% 線性化：log(y) = log(a) + b*log(x)
p = polyfit(log(x), log(y), 1);
b = p(1); a = exp(p(2));

% 使用 lsqcurvefit 的一般非線性擬合
model = @(p, x) p(1)*exp(-p(2)*x);  % 範例：a*exp(-b*x)
p0 = [1, 1];                        % 初始猜測
p = lsqcurvefit(model, p0, xdata, ydata);
```
