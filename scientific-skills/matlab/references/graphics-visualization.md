# 圖形與視覺化參考

## 目錄
1. [2D 繪圖](#2d-繪圖)
2. [3D 繪圖](#3d-繪圖)
3. [特殊圖表](#特殊圖表)
4. [圖形管理](#圖形管理)
5. [自訂設定](#自訂設定)
6. [匯出與儲存](#匯出與儲存)

## 2D 繪圖

### 線圖

```matlab
% 基本線圖
plot(y);                        % 繪製 y 對索引
plot(x, y);                     % 繪製 y 對 x
plot(x, y, 'r-');               % 紅色實線
plot(x, y, 'b--o');             % 藍色虛線帶圓圈

% 線條規格：[顏色][標記][線型]
% 顏色：r g b c m y k w（紅、綠、藍、青、洋紅、黃、黑、白）
% 標記：o + * . x s d ^ v > < p h
% 線條：- -- : -.

% 多資料集
plot(x1, y1, x2, y2, x3, y3);
plot(x, [y1; y2; y3]');         % 欄作為獨立線條

% 帶屬性
plot(x, y, 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
plot(x, y, 'Marker', 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% 取得控制代碼以便稍後修改
h = plot(x, y);
h.LineWidth = 2;
h.Color = 'red';
```

### 散佈圖

```matlab
scatter(x, y);                  % 基本散佈圖
scatter(x, y, sz);              % 帶標記大小
scatter(x, y, sz, c);           % 帶顏色
scatter(x, y, sz, c, 'filled'); % 填滿標記

% sz：純量或向量（標記大小）
% c：顏色規格、純量、向量（色彩對應表）或 RGB 矩陣

% 屬性
scatter(x, y, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'r');
```

### 長條圖

```matlab
bar(y);                         % 垂直長條
bar(x, y);                      % 在指定 x 位置
barh(y);                        % 水平長條

% 分組和堆疊
bar(Y);                         % 每欄是一組
bar(Y, 'stacked');              % 堆疊長條

% 屬性
bar(y, 'FaceColor', 'b', 'EdgeColor', 'k', 'LineWidth', 1.5);
bar(y, 0.5);                    % 長條寬度（0 到 1）
```

### 區域圖

```matlab
area(y);                        % 曲線下填滿區域
area(x, y);
area(Y);                        % 堆疊區域
area(Y, 'FaceAlpha', 0.5);      % 透明
```

### 直方圖

```matlab
histogram(x);                   % 自動分組
histogram(x, nbins);            % 分組數量
histogram(x, edges);            % 指定邊界
histogram(x, 'BinWidth', w);    % 分組寬度

% 正規化
histogram(x, 'Normalization', 'probability');
histogram(x, 'Normalization', 'pdf');
histogram(x, 'Normalization', 'count');  % 預設

% 2D 直方圖
histogram2(x, y);
histogram2(x, y, 'DisplayStyle', 'tile');
histogram2(x, y, 'FaceColor', 'flat');
```

### 誤差長條圖

```matlab
errorbar(x, y, err);            % 對稱誤差
errorbar(x, y, neg, pos);       % 非對稱誤差
errorbar(x, y, yneg, ypos, xneg, xpos);  % X 和 Y 誤差

% 水平
errorbar(x, y, err, 'horizontal');

% 帶線條樣式
errorbar(x, y, err, 'o-', 'LineWidth', 1.5);
```

### 對數圖

```matlab
semilogy(x, y);                 % 對數 y 軸
semilogx(x, y);                 % 對數 x 軸
loglog(x, y);                   % 兩軸都是對數
```

### 極座標圖

```matlab
polarplot(theta, rho);          % 極座標
polarplot(theta, rho, 'r-o');   % 帶線條規格

% 自訂極座標軸
pax = polaraxes;
pax.ThetaDir = 'clockwise';
pax.ThetaZeroLocation = 'top';
```

## 3D 繪圖

### 線條與散佈

```matlab
% 3D 線圖
plot3(x, y, z);
plot3(x, y, z, 'r-', 'LineWidth', 2);

% 3D 散佈圖
scatter3(x, y, z);
scatter3(x, y, z, sz, c, 'filled');
```

### 曲面圖

```matlab
% 先建立網格
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X.^2 + Y.^2;

% 曲面圖
surf(X, Y, Z);                  % 帶邊緣的曲面
surf(Z);                        % 使用索引作為 X, Y

% 曲面屬性
surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
surf(X, Y, Z, 'FaceAlpha', 0.5);  % 透明

% 網格圖（線框）
mesh(X, Y, Z);
mesh(X, Y, Z, 'FaceColor', 'none');

% 帶等高線的曲面
surfc(X, Y, Z);
meshc(X, Y, Z);
```

### 等高線圖

```matlab
contour(X, Y, Z);               % 2D 等高線
contour(X, Y, Z, n);            % n 個等高線層級
contour(X, Y, Z, levels);       % 指定層級
contourf(X, Y, Z);              % 填滿等高線

[C, h] = contour(X, Y, Z);
clabel(C, h);                   % 加入標籤

% 3D 等高線
contour3(X, Y, Z);
```

### 其他 3D 圖表

```matlab
% Bar3
bar3(Z);                        % 3D 長條圖
bar3(Z, 'stacked');

% Pie3
pie3(X);                        % 3D 圓餅圖

% 瀑布圖
waterfall(X, Y, Z);             % 類似 mesh 但無背線

% 帶狀圖
ribbon(Y);                      % 3D 帶狀

% Stem3
stem3(x, y, z);                 % 3D 莖圖
```

### 視角與光照

```matlab
% 設定視角
view(az, el);                   % 方位角、仰角
view(2);                        % 俯視（2D 視圖）
view(3);                        % 預設 3D 視圖
view([1 1 1]);                  % 從方向觀看

% 光照
light;                          % 加入光源
light('Position', [1 0 1]);
lighting gouraud;               % 平滑光照
lighting flat;                  % 平面著色
lighting none;                  % 無光照

% 材質屬性
material shiny;
material dull;
material metal;

% 著色
shading flat;                   % 每面一色
shading interp;                 % 內插顏色
shading faceted;                % 帶邊緣（預設）
```

## 特殊圖表

### 統計圖

```matlab
% 箱型圖
boxplot(data);
boxplot(data, groups);          % 分組
boxplot(data, 'Notch', 'on');   % 帶缺口

% 小提琴圖（R2023b+）
violinplot(data);

% 熱圖
heatmap(data);
heatmap(xLabels, yLabels, data);
heatmap(T, 'XVariable', 'Col1', 'YVariable', 'Col2', 'ColorVariable', 'Val');

% 平行座標
parallelplot(data);
```

### 影像顯示

```matlab
% 顯示影像
imshow(img);                    % 自動縮放
imshow(img, []);                % 縮放到完整範圍
imshow(img, [low high]);        % 指定顯示範圍

% 影像作為圖
image(C);                       % 直接索引顏色
imagesc(data);                  % 縮放顏色
imagesc(data, [cmin cmax]);     % 指定顏色限制

% imagesc 的色彩對應表
imagesc(data);
colorbar;
colormap(jet);
```

### 向量場與流線

```matlab
% 向量場
[X, Y] = meshgrid(-2:0.5:2);
U = -Y;
V = X;
quiver(X, Y, U, V);             % 2D 箭頭
quiver3(X, Y, Z, U, V, W);      % 3D 箭頭

% 流線
streamline(X, Y, U, V, startx, starty);
```

### 圓餅圖與甜甜圈圖

```matlab
pie(X);                         % 圓餅圖
pie(X, explode);                % 突出扇區（邏輯值）
pie(X, labels);                 % 帶標籤

% 甜甜圈圖（使用 patch 或變通方法）
pie(X);
% 在中心加入白色圓圈以產生甜甜圈效果
```

## 圖形管理

### 建立圖形

```matlab
figure;                         % 新圖形視窗
figure(n);                      % 編號為 n 的圖形
fig = figure;                   % 取得控制代碼
fig = figure('Name', 'My Figure', 'Position', [100 100 800 600]);

% 圖形屬性
fig.Color = 'white';
fig.Units = 'pixels';
fig.Position = [left bottom width height];
```

### 子圖

```matlab
subplot(m, n, p);               % m×n 網格，位置 p
subplot(2, 2, 1);               % 2×2 的左上角

% 跨越多個位置
subplot(2, 2, [1 2]);           % 頂列

% 帶間距控制
tiledlayout(2, 2);              % 現代替代方案
nexttile;
plot(x1, y1);
nexttile;
plot(x2, y2);

% 磚塊跨越
nexttile([1 2]);                % 跨越 2 欄
```

### 保持與疊加

```matlab
hold on;                        % 保留現有，加入新圖
plot(x1, y1);
plot(x2, y2);
hold off;                       % 釋放

% 替代方法
hold(ax, 'on');
hold(ax, 'off');
```

### 多軸

```matlab
% 兩個 y 軸
yyaxis left;
plot(x, y1);
ylabel('左 Y');
yyaxis right;
plot(x, y2);
ylabel('右 Y');

% 連結軸
ax1 = subplot(2,1,1); plot(x, y1);
ax2 = subplot(2,1,2); plot(x, y2);
linkaxes([ax1, ax2], 'x');      % 連結 x 軸
```

### 目前物件

```matlab
gcf;                            % 目前圖形控制代碼
gca;                            % 目前座標軸控制代碼
gco;                            % 目前物件控制代碼

% 設定目前
figure(fig);
axes(ax);
```

## 自訂設定

### 標籤與標題

```matlab
title('我的標題');
title('我的標題', 'FontSize', 14, 'FontWeight', 'bold');

xlabel('X 標籤');
ylabel('Y 標籤');
zlabel('Z 標籤');              % 3D 用

% 帶直譯器
title('$$\int_0^1 x^2 dx$$', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'none');
```

### 圖例

```matlab
legend('系列 1', '系列 2');
legend({'系列 1', '系列 2'});
legend('Location', 'best');     % 自動放置
legend('Location', 'northeast');
legend('Location', 'northeastoutside');

% 指定特定圖
h1 = plot(x1, y1);
h2 = plot(x2, y2);
legend([h1, h2], {'資料 1', '資料 2'});

legend('off');                  % 移除圖例
legend('boxoff');               % 移除框線
```

### 軸控制

```matlab
axis([xmin xmax ymin ymax]);    % 設定限制
axis([xmin xmax ymin ymax zmin zmax]);  % 3D
xlim([xmin xmax]);
ylim([ymin ymax]);
zlim([zmin zmax]);

axis equal;                     % 等比例
axis square;                    % 正方形軸
axis tight;                     % 適合資料
axis auto;                      % 自動
axis off;                       % 隱藏軸
axis on;                        % 顯示軸

% 反轉方向
set(gca, 'YDir', 'reverse');
set(gca, 'XDir', 'reverse');
```

### 網格與框線

```matlab
grid on;
grid off;
grid minor;                     % 次要網格線

box on;                         % 顯示框線
box off;                        % 隱藏框線
```

### 刻度

```matlab
xticks([0 1 2 3 4 5]);
yticks(0:0.5:3);

xticklabels({'A', 'B', 'C', 'D', 'E', 'F'});
yticklabels({'低', '中', '高'});

xtickangle(45);                 % 旋轉標籤
ytickformat('%.2f');            % 格式
xtickformat('usd');             % 貨幣
```

### 顏色與色彩對應表

```matlab
% 預定義色彩對應表
colormap(jet);
colormap(parula);               % 預設
colormap(hot);
colormap(cool);
colormap(gray);
colormap(bone);
colormap(hsv);
colormap(turbo);
colormap(viridis);

% 色彩列
colorbar;
colorbar('Location', 'eastoutside');
caxis([cmin cmax]);             % 顏色限制
clim([cmin cmax]);              % R2022a+ 語法

% 自訂色彩對應表
cmap = [1 0 0; 0 1 0; 0 0 1];   % 紅、綠、藍
colormap(cmap);

% 線條顏色順序
colororder(colors);             % R2019b+
```

### 文字與註解

```matlab
% 加入文字
text(x, y, '標籤');
text(x, y, z, '標籤');         % 3D
text(x, y, '標籤', 'FontSize', 12, 'Color', 'red');
text(x, y, '標籤', 'HorizontalAlignment', 'center');

% 註解
annotation('arrow', [x1 x2], [y1 y2]);
annotation('textarrow', [x1 x2], [y1 y2], 'String', '峰值');
annotation('ellipse', [x y w h]);
annotation('rectangle', [x y w h]);
annotation('line', [x1 x2], [y1 y2]);

% 帶 LaTeX 的文字
text(x, y, '$$\alpha = \beta^2$$', 'Interpreter', 'latex');
```

### 線條與形狀

```matlab
% 參考線
xline(5);                       % 在 x=5 的垂直線
yline(10);                      % 在 y=10 的水平線
xline(5, '--r', '閾值');        % 帶標籤

% 形狀
rectangle('Position', [x y w h]);
rectangle('Position', [x y w h], 'Curvature', [0.2 0.2]);  % 圓角

% 補丁（填滿多邊形）
patch(xv, yv, 'blue');
patch(xv, yv, zv, 'blue');      % 3D
```

## 匯出與儲存

### 儲存圖形

```matlab
saveas(gcf, 'figure.png');
saveas(gcf, 'figure.fig');      % MATLAB 圖形檔案
saveas(gcf, 'figure.pdf');
saveas(gcf, 'figure.eps');
```

### Print 命令

```matlab
print('-dpng', 'figure.png');
print('-dpng', '-r300', 'figure.png');  % 300 DPI
print('-dpdf', 'figure.pdf');
print('-dsvg', 'figure.svg');
print('-deps', 'figure.eps');
print('-depsc', 'figure.eps');  % 彩色 EPS

% 發表用向量格式
print('-dpdf', '-painters', 'figure.pdf');
print('-dsvg', '-painters', 'figure.svg');
```

### 匯出圖形（R2020a+）

```matlab
exportgraphics(gcf, 'figure.png');
exportgraphics(gcf, 'figure.png', 'Resolution', 300);
exportgraphics(gcf, 'figure.pdf', 'ContentType', 'vector');
exportgraphics(gca, 'axes_only.png');  % 僅座標軸

% 用於簡報/文件
exportgraphics(gcf, 'figure.emf');    % Windows
exportgraphics(gcf, 'figure.eps');    % LaTeX
```

### 複製到剪貼簿

```matlab
copygraphics(gcf);              % 複製目前圖形
copygraphics(gca);              % 複製目前座標軸
copygraphics(gcf, 'ContentType', 'vector');
```

### 紙張大小（用於列印）

```matlab
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 6 4]);
set(gcf, 'PaperSize', [6 4]);
set(gcf, 'PaperPositionMode', 'auto');
```
