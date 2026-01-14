# PyTorch Geometric 神經網路層參考

本文件提供 `torch_geometric.nn` 中所有可用神經網路層的完整參考。

## 層功能標記

選擇層時，請考慮以下功能標記：

- **SparseTensor**：支援 `torch_sparse.SparseTensor` 格式以進行高效稀疏運算
- **edge_weight**：處理一維邊權重資料
- **edge_attr**：處理多維邊特徵資訊
- **Bipartite**：適用於二分圖（不同的來源/目標節點維度）
- **Static**：對具有批次節點特徵的靜態圖進行運算
- **Lazy**：允許不指定輸入通道維度進行初始化

## 卷積層

### 標準圖卷積

**GCNConv** - 圖卷積網路層
- 實作具有對稱正規化的頻譜圖卷積
- 支援：SparseTensor、edge_weight、Bipartite、Lazy
- 用途：引用網路、社交網路、一般圖學習
- 範例：`GCNConv(in_channels, out_channels, improved=False, cached=True)`

**SAGEConv** - GraphSAGE 層
- 透過鄰域取樣和聚合進行歸納學習
- 支援：SparseTensor、Bipartite、Lazy
- 用途：大型圖、歸納學習、異質特徵
- 範例：`SAGEConv(in_channels, out_channels, aggr='mean')`

**GATConv** - 圖注意力網路層
- 多頭注意力機制，用於自適應鄰居加權
- 支援：SparseTensor、edge_attr、Bipartite、Static、Lazy
- 用途：需要可變鄰居重要性的任務
- 範例：`GATConv(in_channels, out_channels, heads=8, dropout=0.6)`

**GraphConv** - 簡單圖卷積（Morris 等人）
- 具有可選邊權重的基本訊息傳遞
- 支援：SparseTensor、edge_weight、Bipartite、Lazy
- 用途：基準模型、簡單圖結構
- 範例：`GraphConv(in_channels, out_channels, aggr='add')`

**GINConv** - 圖同構網路層
- 用於圖同構測試的最大功率 GNN
- 支援：Bipartite
- 用途：圖分類、分子性質預測
- 範例：`GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU()))`

**TransformerConv** - 圖 Transformer 層
- 將圖結構與 Transformer 注意力結合
- 支援：SparseTensor、Bipartite、Lazy
- 用途：長程依賴、複雜圖
- 範例：`TransformerConv(in_channels, out_channels, heads=8, beta=True)`

**ChebConv** - Chebyshev 頻譜圖卷積
- 使用 Chebyshev 多項式進行高效頻譜濾波
- 支援：SparseTensor、edge_weight、Bipartite、Lazy
- 用途：頻譜圖學習、高效卷積
- 範例：`ChebConv(in_channels, out_channels, K=3)`

**SGConv** - 簡化圖卷積
- 預計算固定數量的傳播步驟
- 支援：SparseTensor、edge_weight、Bipartite、Lazy
- 用途：快速訓練、淺層模型
- 範例：`SGConv(in_channels, out_channels, K=2)`

**APPNP** - 近似個人化神經預測傳播
- 將特徵轉換與傳播分離
- 支援：SparseTensor、edge_weight、Lazy
- 用途：深度傳播而不過度平滑
- 範例：`APPNP(K=10, alpha=0.1)`

**ARMAConv** - ARMA 圖卷積
- 使用 ARMA 濾波器進行圖濾波
- 支援：SparseTensor、edge_weight、Bipartite、Lazy
- 用途：進階頻譜方法
- 範例：`ARMAConv(in_channels, out_channels, num_stacks=3, num_layers=2)`

**GATv2Conv** - 改進版圖注意力網路
- 修正 GAT 中的靜態注意力計算問題
- 支援：SparseTensor、edge_attr、Bipartite、Static、Lazy
- 用途：比原始 GAT 更好的注意力學習
- 範例：`GATv2Conv(in_channels, out_channels, heads=8)`

**SuperGATConv** - 自監督圖注意力
- 添加自監督注意力機制
- 支援：SparseTensor、edge_attr、Bipartite、Static、Lazy
- 用途：自監督學習、有限標籤
- 範例：`SuperGATConv(in_channels, out_channels, heads=8)`

**GMMConv** - 高斯混合模型卷積
- 在偽座標空間中使用高斯核
- 支援：Bipartite
- 用途：點雲、空間資料
- 範例：`GMMConv(in_channels, out_channels, dim=3, kernel_size=5)`

**SplineConv** - 基於樣條的卷積
- B 樣條基函數用於空間濾波
- 支援：Bipartite
- 用途：不規則網格、連續空間
- 範例：`SplineConv(in_channels, out_channels, dim=2, kernel_size=5)`

**NNConv** - 神經網路卷積
- 由神經網路處理邊特徵
- 支援：edge_attr、Bipartite
- 用途：豐富的邊特徵、分子圖
- 範例：`NNConv(in_channels, out_channels, nn=edge_nn, aggr='mean')`

**CGConv** - 晶體圖卷積
- 專為晶體材料設計
- 支援：Bipartite
- 用途：材料科學、晶體結構
- 範例：`CGConv(in_channels, dim=3, batch_norm=True)`

**EdgeConv** - 邊卷積（動態圖 CNN）
- 基於特徵空間動態計算邊
- 支援：Static
- 用途：點雲、動態圖
- 範例：`EdgeConv(nn=edge_nn, aggr='max')`

**PointNetConv** - PointNet++ 卷積
- 用於點雲的局部和全域特徵學習
- 用途：3D 點雲處理
- 範例：`PointNetConv(local_nn, global_nn)`

**ResGatedGraphConv** - 殘差門控圖卷積
- 具有殘差連接的門控機制
- 支援：edge_attr、Bipartite、Lazy
- 用途：深度 GNN、複雜特徵
- 範例：`ResGatedGraphConv(in_channels, out_channels)`

**GENConv** - 廣義圖卷積
- 廣義化多種 GNN 變體
- 支援：SparseTensor、edge_weight、edge_attr、Bipartite、Lazy
- 用途：靈活的架構探索
- 範例：`GENConv(in_channels, out_channels, aggr='softmax', num_layers=2)`

**FiLMConv** - 特徵級線性調制
- 以全域特徵為條件
- 支援：Bipartite、Lazy
- 用途：條件生成、多任務學習
- 範例：`FiLMConv(in_channels, out_channels, num_relations=5)`

**PANConv** - 路徑注意力網路
- 多跳路徑上的注意力
- 支援：SparseTensor、Lazy
- 用途：複雜連接模式
- 範例：`PANConv(in_channels, out_channels, filter_size=3)`

**ClusterGCNConv** - Cluster-GCN 卷積
- 透過圖聚類進行高效訓練
- 支援：edge_attr、Lazy
- 用途：超大型圖
- 範例：`ClusterGCNConv(in_channels, out_channels)`

**MFConv** - 多尺度特徵卷積
- 在多個尺度上聚合特徵
- 支援：SparseTensor、Lazy
- 用途：多尺度模式
- 範例：`MFConv(in_channels, out_channels)`

**RGCNConv** - 關係圖卷積
- 處理多種邊類型
- 支援：SparseTensor、edge_weight、Lazy
- 用途：知識圖譜、異質圖
- 範例：`RGCNConv(in_channels, out_channels, num_relations=10)`

**FAConv** - 頻率自適應卷積
- 頻譜域中的自適應濾波
- 支援：SparseTensor、Lazy
- 用途：頻譜圖學習
- 範例：`FAConv(in_channels, eps=0.1, dropout=0.5)`

### 分子和 3D 卷積

**SchNet** - 連續濾波器卷積層
- 專為分子動力學設計
- 用途：分子性質預測、3D 分子
- 範例：`SchNet(hidden_channels=128, num_filters=64, num_interactions=6)`

**DimeNet** - 方向性訊息傳遞
- 使用方向資訊和角度
- 用途：3D 分子結構、化學性質
- 範例：`DimeNet(hidden_channels=128, out_channels=1, num_blocks=6)`

**PointTransformerConv** - 點雲 Transformer
- 用於 3D 點雲的 Transformer
- 用途：3D 視覺、點雲分割
- 範例：`PointTransformerConv(in_channels, out_channels)`

### 超圖卷積

**HypergraphConv** - 超圖卷積
- 在超邊（連接多個節點的邊）上運算
- 支援：Lazy
- 用途：多路關係、化學反應
- 範例：`HypergraphConv(in_channels, out_channels)`

**HGTConv** - 異質圖 Transformer
- 用於具有多種類型的異質圖的 Transformer
- 支援：Lazy
- 用途：異質網路、知識圖譜
- 範例：`HGTConv(in_channels, out_channels, metadata, heads=8)`

## 聚合運算子

**Aggr** - 基礎聚合類別
- 跨節點的靈活聚合

**SumAggregation** - 求和聚合
- 範例：`SumAggregation()`

**MeanAggregation** - 平均聚合
- 範例：`MeanAggregation()`

**MaxAggregation** - 最大聚合
- 範例：`MaxAggregation()`

**SoftmaxAggregation** - Softmax 加權聚合
- 可學習的注意力權重
- 範例：`SoftmaxAggregation(learn=True)`

**PowerMeanAggregation** - 冪平均聚合
- 可學習的冪參數
- 範例：`PowerMeanAggregation(learn=True)`

**LSTMAggregation** - 基於 LSTM 的聚合
- 鄰居的序列處理
- 範例：`LSTMAggregation(in_channels, out_channels)`

**SetTransformerAggregation** - Set Transformer 聚合
- 用於排列不變聚合的 Transformer
- 範例：`SetTransformerAggregation(in_channels, out_channels)`

**MultiAggregation** - 多重聚合
- 結合多種聚合方法
- 範例：`MultiAggregation(['mean', 'max', 'std'])`

## 池化層

### 全域池化

**global_mean_pool** - 全域平均池化
- 對每個圖的節點特徵取平均
- 範例：`global_mean_pool(x, batch)`

**global_max_pool** - 全域最大池化
- 對每個圖的節點特徵取最大值
- 範例：`global_max_pool(x, batch)`

**global_add_pool** - 全域求和池化
- 對每個圖的節點特徵求和
- 範例：`global_add_pool(x, batch)`

**global_sort_pool** - 全域排序池化
- 排序並串接前 k 個節點
- 範例：`global_sort_pool(x, batch, k=30)`

**GlobalAttention** - 全域注意力池化
- 用於聚合的可學習注意力權重
- 範例：`GlobalAttention(gate_nn)`

**Set2Set** - Set2Set 池化
- 基於 LSTM 的注意力機制
- 範例：`Set2Set(in_channels, processing_steps=3)`

### 分層池化

**TopKPooling** - Top-k 池化
- 基於投影分數保留前 k 個節點
- 範例：`TopKPooling(in_channels, ratio=0.5)`

**SAGPooling** - 自注意力圖池化
- 使用自注意力進行節點選擇
- 範例：`SAGPooling(in_channels, ratio=0.5)`

**ASAPooling** - 自適應結構感知池化
- 結構感知的節點選擇
- 範例：`ASAPooling(in_channels, ratio=0.5)`

**PANPooling** - 路徑注意力池化
- 用於池化的路徑注意力
- 範例：`PANPooling(in_channels, ratio=0.5)`

**EdgePooling** - 邊收縮池化
- 透過收縮邊進行池化
- 範例：`EdgePooling(in_channels)`

**MemPooling** - 基於記憶的池化
- 可學習的聚類分配
- 範例：`MemPooling(in_channels, out_channels, heads=4, num_clusters=10)`

**avg_pool** / **max_pool** - 使用聚類的平均/最大池化
- 在聚類內對節點進行池化
- 範例：`avg_pool(cluster, data)`

## 正規化層

**BatchNorm** - 批次正規化
- 跨批次正規化特徵
- 範例：`BatchNorm(in_channels)`

**LayerNorm** - 層正規化
- 每個樣本正規化特徵
- 範例：`LayerNorm(in_channels)`

**InstanceNorm** - 實例正規化
- 每個樣本和圖正規化
- 範例：`InstanceNorm(in_channels)`

**GraphNorm** - 圖正規化
- 圖特定的正規化
- 範例：`GraphNorm(in_channels)`

**PairNorm** - 配對正規化
- 防止深度 GNN 中的過度平滑
- 範例：`PairNorm(scale_individually=False)`

**MessageNorm** - 訊息正規化
- 在傳遞過程中正規化訊息
- 範例：`MessageNorm(learn_scale=True)`

**DiffGroupNorm** - 可微分群組正規化
- 用於正規化的可學習分組
- 範例：`DiffGroupNorm(in_channels, groups=10)`

## 模型架構

### 預建模型

**GCN** - 完整的圖卷積網路
- 具有 dropout 的多層 GCN
- 範例：`GCN(in_channels, hidden_channels, num_layers, out_channels)`

**GraphSAGE** - 完整的 GraphSAGE 模型
- 具有 dropout 的多層 SAGE
- 範例：`GraphSAGE(in_channels, hidden_channels, num_layers, out_channels)`

**GIN** - 完整的圖同構網路
- 用於圖分類的多層 GIN
- 範例：`GIN(in_channels, hidden_channels, num_layers, out_channels)`

**GAT** - 完整的圖注意力網路
- 具有注意力的多層 GAT
- 範例：`GAT(in_channels, hidden_channels, num_layers, out_channels, heads=8)`

**PNA** - 主鄰域聚合
- 結合多種聚合器和縮放器
- 範例：`PNA(in_channels, hidden_channels, num_layers, out_channels)`

**EdgeCNN** - 邊卷積 CNN
- 用於點雲的動態圖 CNN
- 範例：`EdgeCNN(out_channels, num_layers=3, k=20)`

### 自編碼器

**GAE** - 圖自編碼器
- 將圖編碼到潛在空間
- 範例：`GAE(encoder)`

**VGAE** - 變分圖自編碼器
- 機率圖編碼
- 範例：`VGAE(encoder)`

**ARGA** - 對抗正則化圖自編碼器
- 具有對抗正則化的 GAE
- 範例：`ARGA(encoder, discriminator)`

**ARGVA** - 對抗正則化變分圖自編碼器
- 具有對抗正則化的 VGAE
- 範例：`ARGVA(encoder, discriminator)`

### 知識圖譜嵌入

**TransE** - 平移嵌入
- 學習實體和關係嵌入
- 範例：`TransE(num_nodes, num_relations, hidden_channels)`

**RotatE** - 旋轉嵌入
- 複數空間中的嵌入
- 範例：`RotatE(num_nodes, num_relations, hidden_channels)`

**ComplEx** - 複數嵌入
- 複數值嵌入
- 範例：`ComplEx(num_nodes, num_relations, hidden_channels)`

**DistMult** - 雙線性對角模型
- 簡化的雙線性模型
- 範例：`DistMult(num_nodes, num_relations, hidden_channels)`

## 實用層

**Sequential** - 順序容器
- 串接多個層
- 範例：`Sequential('x, edge_index', [(GCNConv(16, 64), 'x, edge_index -> x'), nn.ReLU()])`

**JumpingKnowledge** - 跳躍知識連接
- 結合所有層的表示
- 模式：'cat'、'max'、'lstm'
- 範例：`JumpingKnowledge(mode='cat')`

**DeepGCNLayer** - 深度 GCN 層包裝器
- 透過跳躍連接實現非常深的 GNN
- 範例：`DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1)`

**MLP** - 多層感知器
- 標準前饋網路
- 範例：`MLP([in_channels, 64, 64, out_channels], dropout=0.5)`

**Linear** - 惰性線性層
- 具有惰性初始化的線性轉換
- 範例：`Linear(in_channels, out_channels, bias=True)`

## 密集層

用於密集（非稀疏）圖表示：

**DenseGCNConv** - 密集 GCN 層
**DenseSAGEConv** - 密集 SAGE 層
**DenseGINConv** - 密集 GIN 層
**DenseGraphConv** - 密集圖卷積

這些在處理小型、全連接或密集表示的圖時很有用。

## 使用提示

1. **從簡單開始**：大多數任務從 GCNConv 或 GATConv 開始
2. **考慮資料類型**：3D 結構使用分子層（SchNet、DimeNet）
3. **檢查功能**：將層功能與您的資料匹配（邊特徵、二分圖等）
4. **深度網路**：深度 GNN 使用正規化（PairNorm、LayerNorm）和 JumpingKnowledge
5. **大型圖**：使用可擴展層（SAGE、Cluster-GCN）搭配鄰居取樣
6. **異質**：使用 RGCNConv、HGTConv 或 to_hetero() 轉換
7. **惰性初始化**：當輸入維度變化或未知時使用惰性層

## 常見模式

### 基本 GNN
```python
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)
```

### 具有正規化的深度 GNN
```python
class DeepGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.jk = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index, batch):
        xs = []
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            xs.append(x)

        x = self.convs[-1](x, edge_index)
        xs.append(x)

        x = self.jk(xs)
        return global_mean_pool(x, batch)
```
