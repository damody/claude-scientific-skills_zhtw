# PyTorch Geometric 轉換參考

本文件提供 `torch_geometric.transforms` 中所有可用轉換的完整參考。

## 概述

轉換（Transforms）在訓練前或訓練期間修改 `Data` 或 `HeteroData` 物件。透過以下方式應用它們：

```python
# 在資料集載入期間
dataset = MyDataset(root='/tmp', transform=MyTransform())

# 應用於個別資料
transform = MyTransform()
data = transform(data)

# 組合多個轉換
from torch_geometric.transforms import Compose
transform = Compose([Transform1(), Transform2(), Transform3()])
```

## 一般轉換

### NormalizeFeatures
**用途**：將節點特徵按列正規化，使其總和為 1
**使用案例**：特徵縮放、類機率特徵
```python
from torch_geometric.transforms import NormalizeFeatures
transform = NormalizeFeatures()
```

### ToDevice
**用途**：將資料傳輸到指定設備（CPU/GPU）
**使用案例**：GPU 訓練、設備管理
```python
from torch_geometric.transforms import ToDevice
transform = ToDevice('cuda')
```

### RandomNodeSplit
**用途**：建立訓練/驗證/測試節點遮罩
**使用案例**：節點分類分割
**參數**：`split='train_rest'`、`num_splits`、`num_val`、`num_test`
```python
from torch_geometric.transforms import RandomNodeSplit
transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
```

### RandomLinkSplit
**用途**：建立訓練/驗證/測試邊分割
**使用案例**：連結預測
**參數**：`num_val`、`num_test`、`is_undirected`、`split_labels`
```python
from torch_geometric.transforms import RandomLinkSplit
transform = RandomLinkSplit(num_val=0.1, num_test=0.2)
```

### IndexToMask
**用途**：將索引轉換為布林遮罩
**使用案例**：資料預處理
```python
from torch_geometric.transforms import IndexToMask
transform = IndexToMask()
```

### MaskToIndex
**用途**：將布林遮罩轉換為索引
**使用案例**：資料預處理
```python
from torch_geometric.transforms import MaskToIndex
transform = MaskToIndex()
```

### FixedPoints
**用途**：取樣固定數量的點
**使用案例**：點雲子取樣
**參數**：`num`、`replace`、`allow_duplicates`
```python
from torch_geometric.transforms import FixedPoints
transform = FixedPoints(1024)
```

### ToDense
**用途**：轉換為密集鄰接矩陣
**使用案例**：小型圖、密集運算
```python
from torch_geometric.transforms import ToDense
transform = ToDense(num_nodes=100)
```

### ToSparseTensor
**用途**：將 edge_index 轉換為 SparseTensor
**使用案例**：高效稀疏運算
**參數**：`remove_edge_index`、`fill_cache`
```python
from torch_geometric.transforms import ToSparseTensor
transform = ToSparseTensor()
```

## 圖結構轉換

### ToUndirected
**用途**：將有向圖轉換為無向圖
**使用案例**：無向圖演算法
**參數**：`reduce='add'`（如何處理重複邊）
```python
from torch_geometric.transforms import ToUndirected
transform = ToUndirected()
```

### AddSelfLoops
**用途**：為所有節點添加自環
**使用案例**：GCN 風格卷積
**參數**：`fill_value`（自環的邊屬性）
```python
from torch_geometric.transforms import AddSelfLoops
transform = AddSelfLoops()
```

### RemoveSelfLoops
**用途**：移除所有自環
**使用案例**：清理圖結構
```python
from torch_geometric.transforms import RemoveSelfLoops
transform = RemoveSelfLoops()
```

### RemoveIsolatedNodes
**用途**：移除沒有邊的節點
**使用案例**：圖清理
```python
from torch_geometric.transforms import RemoveIsolatedNodes
transform = RemoveIsolatedNodes()
```

### RemoveDuplicatedEdges
**用途**：移除重複邊
**使用案例**：圖清理
```python
from torch_geometric.transforms import RemoveDuplicatedEdges
transform = RemoveDuplicatedEdges()
```

### LargestConnectedComponents
**用途**：僅保留最大連通分量
**使用案例**：聚焦於主要圖結構
**參數**：`num_components`（要保留的分量數量）
```python
from torch_geometric.transforms import LargestConnectedComponents
transform = LargestConnectedComponents(num_components=1)
```

### KNNGraph
**用途**：基於 k 近鄰建立邊
**使用案例**：點雲、空間資料
**參數**：`k`、`loop`、`force_undirected`、`flow`
```python
from torch_geometric.transforms import KNNGraph
transform = KNNGraph(k=6)
```

### RadiusGraph
**用途**：在半徑內建立邊
**使用案例**：點雲、空間資料
**參數**：`r`、`loop`、`max_num_neighbors`、`flow`
```python
from torch_geometric.transforms import RadiusGraph
transform = RadiusGraph(r=0.1)
```

### Delaunay
**用途**：計算 Delaunay 三角剖分
**使用案例**：2D/3D 空間圖
```python
from torch_geometric.transforms import Delaunay
transform = Delaunay()
```

### FaceToEdge
**用途**：將網格面轉換為邊
**使用案例**：網格處理
```python
from torch_geometric.transforms import FaceToEdge
transform = FaceToEdge()
```

### LineGraph
**用途**：將圖轉換為其線圖
**使用案例**：以邊為中心的分析
**參數**：`force_directed`
```python
from torch_geometric.transforms import LineGraph
transform = LineGraph()
```

### GDC
**用途**：圖擴散卷積預處理
**使用案例**：改進的訊息傳遞
**參數**：`self_loop_weight`、`normalization_in`、`normalization_out`、`diffusion_kwargs`
```python
from torch_geometric.transforms import GDC
transform = GDC(self_loop_weight=1, normalization_in='sym',
                diffusion_kwargs=dict(method='ppr', alpha=0.15))
```

### SIGN
**用途**：可擴展 Inception 圖神經網路預處理
**使用案例**：高效多尺度特徵
**參數**：`K`（跳數）
```python
from torch_geometric.transforms import SIGN
transform = SIGN(K=3)
```

## 特徵轉換

### OneHotDegree
**用途**：對節點度進行 one-hot 編碼
**使用案例**：度作為特徵
**參數**：`max_degree`、`cat`（與現有特徵串接）
```python
from torch_geometric.transforms import OneHotDegree
transform = OneHotDegree(max_degree=100)
```

### LocalDegreeProfile
**用途**：附加局部度分佈
**使用案例**：結構性節點特徵
```python
from torch_geometric.transforms import LocalDegreeProfile
transform = LocalDegreeProfile()
```

### Constant
**用途**：為節點添加常數特徵
**使用案例**：無特徵圖
**參數**：`value`、`cat`
```python
from torch_geometric.transforms import Constant
transform = Constant(value=1.0)
```

### TargetIndegree
**用途**：將入度儲存為目標
**使用案例**：度預測
**參數**：`norm`、`max_value`
```python
from torch_geometric.transforms import TargetIndegree
transform = TargetIndegree(norm=False)
```

### AddRandomWalkPE
**用途**：添加隨機漫步位置編碼
**使用案例**：位置資訊
**參數**：`walk_length`、`attr_name`
```python
from torch_geometric.transforms import AddRandomWalkPE
transform = AddRandomWalkPE(walk_length=20)
```

### AddLaplacianEigenvectorPE
**用途**：添加 Laplacian 特徵向量位置編碼
**使用案例**：頻譜位置資訊
**參數**：`k`（特徵向量數量）、`attr_name`
```python
from torch_geometric.transforms import AddLaplacianEigenvectorPE
transform = AddLaplacianEigenvectorPE(k=10)
```

### AddMetaPaths
**用途**：添加元路徑誘導的邊
**使用案例**：異質圖
**參數**：`metapaths`、`drop_orig_edges`、`drop_unconnected_nodes`
```python
from torch_geometric.transforms import AddMetaPaths
metapaths = [[('author', 'paper'), ('paper', 'author')]]  # 共同作者關係
transform = AddMetaPaths(metapaths)
```

### SVDFeatureReduction
**用途**：透過 SVD 降低特徵維度
**使用案例**：降維
**參數**：`out_channels`
```python
from torch_geometric.transforms import SVDFeatureReduction
transform = SVDFeatureReduction(out_channels=64)
```

## 視覺/空間轉換

### Center
**用途**：將節點位置置中
**使用案例**：點雲預處理
```python
from torch_geometric.transforms import Center
transform = Center()
```

### NormalizeScale
**用途**：將位置正規化到單位球
**使用案例**：點雲正規化
```python
from torch_geometric.transforms import NormalizeScale
transform = NormalizeScale()
```

### NormalizeRotation
**用途**：旋轉到主成分
**使用案例**：旋轉不變學習
**參數**：`max_points`
```python
from torch_geometric.transforms import NormalizeRotation
transform = NormalizeRotation()
```

### Distance
**用途**：將歐氏距離儲存為邊屬性
**使用案例**：空間圖
**參數**：`norm`、`max_value`、`cat`
```python
from torch_geometric.transforms import Distance
transform = Distance(norm=False, cat=False)
```

### Cartesian
**用途**：將相對笛卡爾座標儲存為邊屬性
**使用案例**：空間關係
**參數**：`norm`、`max_value`、`cat`
```python
from torch_geometric.transforms import Cartesian
transform = Cartesian(norm=False)
```

### Polar
**用途**：將極座標儲存為邊屬性
**使用案例**：2D 空間圖
**參數**：`norm`、`max_value`、`cat`
```python
from torch_geometric.transforms import Polar
transform = Polar(norm=False)
```

### Spherical
**用途**：將球座標儲存為邊屬性
**使用案例**：3D 空間圖
**參數**：`norm`、`max_value`、`cat`
```python
from torch_geometric.transforms import Spherical
transform = Spherical(norm=False)
```

### LocalCartesian
**用途**：在局部座標系中儲存座標
**使用案例**：局部空間特徵
**參數**：`norm`、`cat`
```python
from torch_geometric.transforms import LocalCartesian
transform = LocalCartesian()
```

### PointPairFeatures
**用途**：計算點對特徵
**使用案例**：3D 配準、對應
**參數**：`cat`
```python
from torch_geometric.transforms import PointPairFeatures
transform = PointPairFeatures()
```

## 資料增強

### RandomJitter
**用途**：隨機抖動節點位置
**使用案例**：點雲增強
**參數**：`translate`、`scale`
```python
from torch_geometric.transforms import RandomJitter
transform = RandomJitter(0.01)
```

### RandomFlip
**用途**：沿軸隨機翻轉位置
**使用案例**：幾何增強
**參數**：`axis`、`p`（機率）
```python
from torch_geometric.transforms import RandomFlip
transform = RandomFlip(axis=0, p=0.5)
```

### RandomScale
**用途**：隨機縮放位置
**使用案例**：縮放增強
**參數**：`scales`（最小值、最大值）
```python
from torch_geometric.transforms import RandomScale
transform = RandomScale((0.9, 1.1))
```

### RandomRotate
**用途**：隨機旋轉位置
**使用案例**：旋轉增強
**參數**：`degrees`（範圍）、`axis`（旋轉軸）
```python
from torch_geometric.transforms import RandomRotate
transform = RandomRotate(degrees=15, axis=2)
```

### RandomShear
**用途**：隨機剪切位置
**使用案例**：幾何增強
**參數**：`shear`（範圍）
```python
from torch_geometric.transforms import RandomShear
transform = RandomShear(0.1)
```

### RandomTranslate
**用途**：隨機平移位置
**使用案例**：平移增強
**參數**：`translate`（範圍）
```python
from torch_geometric.transforms import RandomTranslate
transform = RandomTranslate(0.1)
```

### LinearTransformation
**用途**：應用線性變換矩陣
**使用案例**：自訂幾何變換
**參數**：`matrix`
```python
from torch_geometric.transforms import LinearTransformation
import torch
matrix = torch.eye(3)
transform = LinearTransformation(matrix)
```

## 網格處理

### SamplePoints
**用途**：從網格均勻取樣點
**使用案例**：網格到點雲轉換
**參數**：`num`、`remove_faces`、`include_normals`
```python
from torch_geometric.transforms import SamplePoints
transform = SamplePoints(num=1024)
```

### GenerateMeshNormals
**用途**：生成面/頂點法線
**使用案例**：網格處理
```python
from torch_geometric.transforms import GenerateMeshNormals
transform = GenerateMeshNormals()
```

### FaceToEdge
**用途**：將網格面轉換為邊
**使用案例**：網格到圖轉換
**參數**：`remove_faces`
```python
from torch_geometric.transforms import FaceToEdge
transform = FaceToEdge()
```

## 取樣和分割

### GridSampling
**用途**：在體素網格中聚類點
**使用案例**：點雲降取樣
**參數**：`size`（體素大小）、`start`、`end`
```python
from torch_geometric.transforms import GridSampling
transform = GridSampling(size=0.1)
```

### FixedPoints
**用途**：取樣固定數量的點
**使用案例**：統一點雲大小
**參數**：`num`、`replace`、`allow_duplicates`
```python
from torch_geometric.transforms import FixedPoints
transform = FixedPoints(num=2048, replace=False)
```

### RandomScale
**用途**：從範圍取樣進行隨機縮放
**使用案例**：縮放增強（已在上方列出）

### VirtualNode
**用途**：添加連接到所有節點的虛擬節點
**使用案例**：全域資訊傳播
```python
from torch_geometric.transforms import VirtualNode
transform = VirtualNode()
```

## 專用轉換

### ToSLIC
**用途**：將圖像轉換為超像素圖（SLIC 演算法）
**使用案例**：圖像作為圖
**參數**：`num_segments`、`compactness`、`add_seg`、`add_img`
```python
from torch_geometric.transforms import ToSLIC
transform = ToSLIC(num_segments=75)
```

### GCNNorm
**用途**：對邊應用 GCN 風格正規化
**使用案例**：GCN 預處理
**參數**：`add_self_loops`
```python
from torch_geometric.transforms import GCNNorm
transform = GCNNorm(add_self_loops=True)
```

### LaplacianLambdaMax
**用途**：計算最大 Laplacian 特徵值
**使用案例**：ChebConv 預處理
**參數**：`normalization`、`is_undirected`
```python
from torch_geometric.transforms import LaplacianLambdaMax
transform = LaplacianLambdaMax(normalization='sym')
```

### NormalizeRotation
**用途**：旋轉網格/點雲以對齊主軸
**使用案例**：標準方位
**參數**：`max_points`
```python
from torch_geometric.transforms import NormalizeRotation
transform = NormalizeRotation()
```

## 組合和應用

### Compose
**用途**：串接多個轉換
**使用案例**：複雜預處理管線
```python
from torch_geometric.transforms import Compose
transform = Compose([
    Center(),
    NormalizeScale(),
    KNNGraph(k=6),
    Distance(norm=False),
])
```

### BaseTransform
**用途**：自訂轉換的基礎類別
**使用案例**：實作自訂轉換
```python
from torch_geometric.transforms import BaseTransform

class MyTransform(BaseTransform):
    def __init__(self, param):
        self.param = param

    def __call__(self, data):
        # 修改資料
        data.x = data.x * self.param
        return data
```

## 常見轉換組合

### 節點分類預處理
```python
transform = Compose([
    NormalizeFeatures(),
    RandomNodeSplit(num_val=0.1, num_test=0.2),
])
```

### 點雲處理
```python
transform = Compose([
    Center(),
    NormalizeScale(),
    RandomRotate(degrees=15, axis=2),
    RandomJitter(0.01),
    KNNGraph(k=6),
    Distance(norm=False),
])
```

### 網格到圖
```python
transform = Compose([
    FaceToEdge(remove_faces=True),
    GenerateMeshNormals(),
    Distance(norm=True),
])
```

### 圖結構增強
```python
transform = Compose([
    ToUndirected(),
    AddSelfLoops(),
    RemoveIsolatedNodes(),
    GCNNorm(),
])
```

### 異質圖預處理
```python
transform = Compose([
    AddMetaPaths(metapaths=[
        [('author', 'paper'), ('paper', 'author')],
        [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]
    ]),
    RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2),
])
```

### 連結預測
```python
transform = Compose([
    NormalizeFeatures(),
    RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True),
])
```

## 使用提示

1. **順序很重要**：在特徵轉換之前應用結構轉換
2. **快取**：某些轉換（如 GDC）計算成本高——只應用一次
3. **增強**：僅在訓練期間使用 Random* 轉換
4. **謹慎組合**：太多轉換會減慢資料載入速度
5. **自訂轉換**：繼承 `BaseTransform` 以實作自訂邏輯
6. **預轉換**：在資料集處理期間應用昂貴的轉換一次：
   ```python
   dataset = MyDataset(root='/tmp', pre_transform=ExpensiveTransform())
   ```
7. **動態轉換**：在訓練期間應用低成本轉換：
   ```python
   dataset = MyDataset(root='/tmp', transform=CheapTransform())
   ```

## 效能考量

**昂貴的轉換**（應用為 pre_transform）：
- GDC
- SIGN
- KNNGraph（對於大型點雲）
- AddLaplacianEigenvectorPE
- SVDFeatureReduction

**低成本轉換**（應用為 transform）：
- NormalizeFeatures
- ToUndirected
- AddSelfLoops
- Random* 增強
- ToDevice

**範例**：
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, GDC, NormalizeFeatures

# 昂貴的預處理只做一次
pre_transform = GDC(
    self_loop_weight=1,
    normalization_in='sym',
    diffusion_kwargs=dict(method='ppr', alpha=0.15)
)

# 每次應用低成本轉換
transform = NormalizeFeatures()

dataset = Planetoid(
    root='/tmp/Cora',
    name='Cora',
    pre_transform=pre_transform,
    transform=transform
)
```
