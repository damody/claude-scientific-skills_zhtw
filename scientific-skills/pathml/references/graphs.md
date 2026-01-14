# 圖構建與空間分析

## 概述

PathML 提供從組織影像構建空間圖的工具，用於表示細胞和組織層級的關係。基於圖的表示能實現複雜的空間分析，包括鄰域分析、細胞間交互作用研究和圖神經網路應用。這些圖同時捕捉形態特徵和空間拓撲結構，用於下游計算分析。

## 圖類型

PathML 支援構建多種圖類型：

### 細胞圖
- 節點代表單個細胞
- 邊代表空間鄰近性或生物交互作用
- 節點特徵包括形態、標記表達、細胞類型
- 適用於單細胞空間分析

### 組織圖
- 節點代表組織區域或超像素
- 邊代表空間鄰接性
- 節點特徵包括組織組成、紋理特徵
- 適用於組織層級空間模式

### 空間轉錄組學圖
- 節點代表空間點或細胞
- 邊編碼空間關係
- 節點特徵包括基因表達譜
- 適用於空間組學分析

## 圖構建工作流程

### 從分割到圖

將細胞核或細胞分割結果轉換為空間圖：

```python
from pathml.graph import CellGraph
from pathml.preprocessing import Pipeline, SegmentMIF
import numpy as np

# 1. 執行細胞分割
pipeline = Pipeline([
    SegmentMIF(
        nuclear_channel='DAPI',
        cytoplasm_channel='CD45',
        model='mesmer'
    )
])
pipeline.run(slide)

# 2. 提取實例分割遮罩
inst_map = slide.masks['cell_segmentation']

# 3. 構建細胞圖
cell_graph = CellGraph.from_instance_map(
    inst_map,
    image=slide.image,  # 可選：用於提取視覺特徵
    connectivity='delaunay',  # 'knn'、'radius' 或 'delaunay'
    k=5,  # 對於 knn：鄰居數量
    radius=50  # 對於 radius：像素距離閾值
)

# 4. 存取圖元件
nodes = cell_graph.nodes  # 節點特徵
edges = cell_graph.edges  # 邊列表
adjacency = cell_graph.adjacency_matrix  # 鄰接矩陣
```

### 連接方法

**K 近鄰（KNN）：**
```python
# 將每個細胞連接到其 k 個最近鄰居
graph = CellGraph.from_instance_map(
    inst_map,
    connectivity='knn',
    k=5  # 鄰居數量
)
```
- 每個節點固定的度數
- 捕捉局部鄰域
- 簡單且可解釋

**基於半徑：**
```python
# 連接距離閾值內的細胞
graph = CellGraph.from_instance_map(
    inst_map,
    connectivity='radius',
    radius=100,  # 像素中的最大距離
    distance_metric='euclidean'  # 或 'manhattan'、'chebyshev'
)
```
- 基於密度的可變度數
- 生物學導向（交互作用範圍）
- 捕捉物理鄰近性

**Delaunay 三角剖分：**
```python
# 使用 Delaunay 三角剖分連接細胞
graph = CellGraph.from_instance_map(
    inst_map,
    connectivity='delaunay'
)
```
- 從空間位置創建連通圖
- 無孤立節點（在凸包內）
- 捕捉空間鑲嵌

**基於接觸：**
```python
# 連接邊界接觸的細胞
graph = CellGraph.from_instance_map(
    inst_map,
    connectivity='contact',
    dilation=2  # 擴張邊界以捕捉近接觸
)
```
- 物理細胞間接觸
- 最直接的生物學意義
- 分離細胞的邊較稀疏

## 節點特徵

### 形態特徵

為每個細胞提取形狀和大小特徵：

```python
from pathml.graph import extract_morphology_features

# 計算形態特徵
morphology_features = extract_morphology_features(
    inst_map,
    features=[
        'area',  # 細胞面積（像素）
        'perimeter',  # 細胞周長
        'eccentricity',  # 形狀伸長度
        'solidity',  # 凸性度量
        'major_axis_length',
        'minor_axis_length',
        'orientation'  # 細胞方向角度
    ]
)

# 添加到圖
cell_graph.add_node_features(morphology_features, feature_names=['area', 'perimeter', ...])
```

**可用形態特徵：**
- **Area** - 像素數量
- **Perimeter** - 邊界長度
- **Eccentricity** - 0（圓形）到 1（線形）
- **Solidity** - 面積 / 凸包面積
- **Circularity** - 4π × 面積 / 周長²
- **Major/Minor axis** - 擬合橢圓的軸長度
- **Orientation** - 主軸角度
- **Extent** - 面積 / 邊界框面積

### 強度特徵

提取標記表達或強度統計：

```python
from pathml.graph import extract_intensity_features

# 提取每個細胞的平均標記強度
intensity_features = extract_intensity_features(
    inst_map,
    image=multichannel_image,  # 形狀：(H, W, C)
    channel_names=['DAPI', 'CD3', 'CD4', 'CD8', 'CD20'],
    statistics=['mean', 'std', 'median', 'max']
)

# 添加到圖
cell_graph.add_node_features(
    intensity_features,
    feature_names=['DAPI_mean', 'CD3_mean', ...]
)
```

**可用統計量：**
- **mean** - 平均強度
- **median** - 中位強度
- **std** - 標準差
- **max** - 最大強度
- **min** - 最小強度
- **quantile_25/75** - 四分位數

### 紋理特徵

為每個細胞區域計算紋理描述子：

```python
from pathml.graph import extract_texture_features

# Haralick 紋理特徵
texture_features = extract_texture_features(
    inst_map,
    image=grayscale_image,
    features='haralick',  # 或 'lbp'、'gabor'
    distance=1,
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
)

cell_graph.add_node_features(texture_features)
```

### 細胞類型註釋

從分類添加細胞類型標籤：

```python
# 來自 ML 模型預測
cell_types = hovernet_type_predictions  # 細胞類型 ID 陣列

cell_graph.add_node_features(
    cell_types,
    feature_names=['cell_type']
)

# 對細胞類型進行獨熱編碼
cell_type_onehot = one_hot_encode(cell_types, num_classes=5)
cell_graph.add_node_features(
    cell_type_onehot,
    feature_names=['type_epithelial', 'type_inflammatory', ...]
)
```

## 邊特徵

### 空間距離

基於空間關係計算邊特徵：

```python
from pathml.graph import compute_edge_distances

# 添加成對距離作為邊特徵
distances = compute_edge_distances(
    cell_graph,
    metric='euclidean'  # 或 'manhattan'、'chebyshev'
)

cell_graph.add_edge_features(distances, feature_names=['distance'])
```

### 交互作用特徵

模擬細胞類型之間的生物交互作用：

```python
from pathml.graph import compute_interaction_features

# 沿邊的細胞類型共現
interaction_features = compute_interaction_features(
    cell_graph,
    cell_types=cell_type_labels,
    interaction_type='categorical'  # 或 'numerical'
)

cell_graph.add_edge_features(interaction_features)
```

## 圖層級特徵

匯總整個圖的特徵：

```python
from pathml.graph import compute_graph_features

# 拓撲特徵
graph_features = compute_graph_features(
    cell_graph,
    features=[
        'num_nodes',
        'num_edges',
        'average_degree',
        'clustering_coefficient',
        'average_path_length',
        'diameter'
    ]
)

# 細胞組成特徵
composition = cell_graph.compute_cell_type_composition(
    cell_type_labels,
    normalize=True  # 比例
)
```

## 空間分析

### 鄰域分析

分析細胞鄰域和微環境：

```python
from pathml.graph import analyze_neighborhoods

# 表徵每個細胞周圍的鄰域
neighborhoods = analyze_neighborhoods(
    cell_graph,
    cell_types=cell_type_labels,
    radius=100,  # 鄰域半徑
    metrics=['diversity', 'density', 'composition']
)

# 鄰域多樣性（Shannon 熵）
diversity = neighborhoods['diversity']

# 每個鄰域中的細胞類型組成
composition = neighborhoods['composition']  # (n_cells, n_cell_types)
```

### 空間聚類

識別細胞類型的空間聚類：

```python
from pathml.graph import spatial_clustering
import matplotlib.pyplot as plt

# 檢測空間聚類
clusters = spatial_clustering(
    cell_graph,
    cell_positions,
    method='dbscan',  # 或 'kmeans'、'hierarchical'
    eps=50,  # DBSCAN：鄰域半徑
    min_samples=10  # DBSCAN：最小聚類大小
)

# 視覺化聚類
plt.scatter(
    cell_positions[:, 0],
    cell_positions[:, 1],
    c=clusters,
    cmap='tab20'
)
plt.title('空間聚類')
plt.show()
```

### 細胞間交互作用分析

測試細胞類型交互作用的富集或耗竭：

```python
from pathml.graph import cell_interaction_analysis

# 測試顯著交互作用
interaction_results = cell_interaction_analysis(
    cell_graph,
    cell_types=cell_type_labels,
    method='permutation',  # 或 'expected'
    n_permutations=1000,
    significance_level=0.05
)

# 交互作用分數（正 = 吸引，負 = 排斥）
interaction_matrix = interaction_results['scores']

# 使用熱力圖視覺化
import seaborn as sns
sns.heatmap(
    interaction_matrix,
    cmap='RdBu_r',
    center=0,
    xticklabels=cell_type_names,
    yticklabels=cell_type_names
)
plt.title('細胞間交互作用分數')
plt.show()
```

### 空間統計

計算空間統計和模式：

```python
from pathml.graph import spatial_statistics

# 用於空間點模式的 Ripley K 函數
ripleys_k = spatial_statistics(
    cell_positions,
    cell_types=cell_type_labels,
    statistic='ripleys_k',
    radii=np.linspace(0, 200, 50)
)

# 最近鄰距離
nn_distances = spatial_statistics(
    cell_positions,
    statistic='nearest_neighbor',
    by_cell_type=True
)
```

## 與圖神經網路整合

### 轉換為 PyTorch Geometric 格式

```python
from pathml.graph import to_pyg
import torch
from torch_geometric.data import Data

# 轉換為 PyTorch Geometric Data 物件
pyg_data = cell_graph.to_pyg()

# 存取元件
x = pyg_data.x  # 節點特徵 (n_nodes, n_features)
edge_index = pyg_data.edge_index  # 邊連接 (2, n_edges)
edge_attr = pyg_data.edge_attr  # 邊特徵 (n_edges, n_edge_features)
y = pyg_data.y  # 圖層級標籤
pos = pyg_data.pos  # 節點位置 (n_nodes, 2)

# 與 PyTorch Geometric 一起使用
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN(in_channels=pyg_data.num_features, hidden_channels=64, out_channels=5)
output = model(pyg_data)
```

### 多個切片的圖資料集

```python
from pathml.graph import GraphDataset
from torch_geometric.loader import DataLoader

# 從多個切片創建圖資料集
graphs = []
for slide in slides:
    # 為每個切片構建圖
    cell_graph = CellGraph.from_instance_map(slide.inst_map, ...)
    pyg_graph = cell_graph.to_pyg()
    graphs.append(pyg_graph)

# 創建 DataLoader
loader = DataLoader(graphs, batch_size=32, shuffle=True)

# 訓練 GNN
for batch in loader:
    output = model(batch)
    loss = criterion(output, batch.y)
    loss.backward()
    optimizer.step()
```

## 視覺化

### 圖視覺化

```python
import matplotlib.pyplot as plt
import networkx as nx

# 轉換為 NetworkX
nx_graph = cell_graph.to_networkx()

# 使用細胞位置作為佈局繪製圖
pos = {i: cell_graph.positions[i] for i in range(len(cell_graph.nodes))}

plt.figure(figsize=(12, 12))
nx.draw_networkx(
    nx_graph,
    pos=pos,
    node_color=cell_type_labels,
    node_size=50,
    cmap='tab10',
    with_labels=False,
    alpha=0.8
)
plt.axis('equal')
plt.title('細胞圖')
plt.show()
```

### 疊加在組織影像上

```python
from pathml.graph import visualize_graph_on_image

# 視覺化疊加在組織上的圖
fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(tissue_image)

# 繪製邊
for edge in cell_graph.edges:
    node1, node2 = edge
    pos1 = cell_graph.positions[node1]
    pos2 = cell_graph.positions[node2]
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b-', alpha=0.3, linewidth=0.5)

# 繪製按類型著色的節點
for cell_type in np.unique(cell_type_labels):
    mask = cell_type_labels == cell_type
    positions = cell_graph.positions[mask]
    ax.scatter(positions[:, 0], positions[:, 1], label=f'類型 {cell_type}', s=20)

ax.legend()
ax.axis('off')
plt.title('組織上的細胞圖')
plt.show()
```

## 完整工作流程範例

```python
from pathml.core import SlideData, CODEXSlide
from pathml.preprocessing import Pipeline, CollapseRunsCODEX, SegmentMIF
from pathml.graph import CellGraph, extract_morphology_features, extract_intensity_features
import matplotlib.pyplot as plt

# 1. 載入並預處理切片
slide = CODEXSlide('path/to/codex', stain='IF')

pipeline = Pipeline([
    CollapseRunsCODEX(z_slice=2),
    SegmentMIF(
        nuclear_channel='DAPI',
        cytoplasm_channel='CD45',
        model='mesmer'
    )
])
pipeline.run(slide)

# 2. 構建細胞圖
inst_map = slide.masks['cell_segmentation']
cell_graph = CellGraph.from_instance_map(
    inst_map,
    image=slide.image,
    connectivity='knn',
    k=6
)

# 3. 提取特徵
# 形態特徵
morph_features = extract_morphology_features(
    inst_map,
    features=['area', 'perimeter', 'eccentricity', 'solidity']
)
cell_graph.add_node_features(morph_features)

# 強度特徵（標記表達）
intensity_features = extract_intensity_features(
    inst_map,
    image=slide.image,
    channel_names=['DAPI', 'CD3', 'CD4', 'CD8', 'CD20'],
    statistics=['mean', 'std']
)
cell_graph.add_node_features(intensity_features)

# 4. 空間分析
from pathml.graph import analyze_neighborhoods

neighborhoods = analyze_neighborhoods(
    cell_graph,
    cell_types=cell_type_predictions,
    radius=100,
    metrics=['diversity', 'composition']
)

# 5. 匯出用於 GNN
pyg_data = cell_graph.to_pyg()

# 6. 視覺化
plt.figure(figsize=(15, 15))
plt.imshow(slide.image)

# 疊加圖
nx_graph = cell_graph.to_networkx()
pos = {i: cell_graph.positions[i] for i in range(cell_graph.num_nodes)}
nx.draw_networkx(
    nx_graph,
    pos=pos,
    node_color=cell_type_predictions,
    cmap='tab10',
    node_size=30,
    with_labels=False
)
plt.axis('off')
plt.title('帶空間鄰域的細胞圖')
plt.show()
```

## 效能考量

**大型組織切片：**
- 逐圖磚構建圖，然後合併
- 使用稀疏鄰接矩陣
- 利用 GPU 進行特徵提取

**記憶體效率：**
- 僅儲存必要的邊特徵
- 使用 int32/float32 代替 int64/float64
- 批次處理多個切片

**計算效率：**
- 跨細胞並行化特徵提取
- 使用 KNN 進行更快的鄰居查詢
- 快取計算的特徵

## 最佳實踐

1. **選擇適當的連接性：** KNN 用於均勻分析，radius 用於物理交互作用，contact 用於直接細胞間通訊

2. **標準化特徵：** 縮放形態和強度特徵以與 GNN 相容

3. **處理邊緣效應：** 排除邊界細胞或使用組織遮罩定義有效區域

4. **驗證圖構建：** 在大規模處理前在小區域上視覺化圖

5. **結合多種特徵類型：** 形態 + 強度 + 紋理提供豐富的表示

6. **考慮組織背景：** 組織類型影響適當的圖參數（連接性、半徑）

## 常見問題與解決方案

**問題：邊太多/太少**
- 調整 k（KNN）或 radius（基於半徑）參數
- 驗證像素到微米的轉換以確保生物學相關性

**問題：大型圖的記憶體錯誤**
- 分別處理圖磚並合併圖
- 使用稀疏矩陣表示
- 將邊特徵減少到必要的

**問題：組織邊界處缺少細胞**
- 應用 edge_correction 參數
- 使用組織遮罩排除無效區域

**問題：特徵縮放不一致**
- 標準化特徵：`(x - mean) / std`
- 對異常值使用穩健縮放

## 其他資源

- **PathML Graph API：** https://pathml.readthedocs.io/en/latest/api_graph_reference.html
- **PyTorch Geometric：** https://pytorch-geometric.readthedocs.io/
- **NetworkX：** https://networkx.org/
- **空間統計：** Baddeley 等人，"Spatial Point Patterns: Methodology and Applications with R"
