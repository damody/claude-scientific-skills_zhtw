# PyTorch Geometric 資料集參考

本文件提供 `torch_geometric.datasets` 中所有可用資料集的完整目錄。

## 引用網路

### Planetoid
**用途**：節點分類（node classification）、半監督學習（semi-supervised learning）
**網路**：Cora、CiteSeer、PubMed
**描述**：引用網路，其中節點是論文，邊是引用關係
- **Cora**：2,708 個節點、5,429 條邊、7 個類別、1,433 個特徵
- **CiteSeer**：3,327 個節點、4,732 條邊、6 個類別、3,703 個特徵
- **PubMed**：19,717 個節點、44,338 條邊、3 個類別、500 個特徵

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

### Coauthor
**用途**：協作網路上的節點分類
**網路**：CS、Physics
**描述**：來自 Microsoft Academic Graph 的共同作者網路
- **CS**：18,333 個節點、81,894 條邊、15 個類別（電腦科學）
- **Physics**：34,493 個節點、247,962 條邊、5 個類別（物理學）

```python
from torch_geometric.datasets import Coauthor
dataset = Coauthor(root='/tmp/CS', name='CS')
```

### Amazon
**用途**：產品網路上的節點分類
**網路**：Computers、Photo
**描述**：Amazon 共同購買網路，其中節點是產品
- **Computers**：13,752 個節點、245,861 條邊、10 個類別
- **Photo**：7,650 個節點、119,081 條邊、8 個類別

```python
from torch_geometric.datasets import Amazon
dataset = Amazon(root='/tmp/Computers', name='Computers')
```

### CitationFull
**用途**：引用網路分析
**網路**：Cora、Cora_ML、DBLP、PubMed
**描述**：未經取樣的完整引用網路

```python
from torch_geometric.datasets import CitationFull
dataset = CitationFull(root='/tmp/Cora', name='Cora')
```

## 圖分類

### TUDataset
**用途**：圖分類（graph classification）、圖核基準測試（graph kernel benchmarks）
**描述**：超過 120 個圖分類資料集的集合
- **MUTAG**：188 個圖、2 個類別（分子化合物）
- **PROTEINS**：1,113 個圖、2 個類別（蛋白質結構）
- **ENZYMES**：600 個圖、6 個類別（蛋白質酵素）
- **IMDB-BINARY**：1,000 個圖、2 個類別（社交網路）
- **REDDIT-BINARY**：2,000 個圖、2 個類別（討論串）
- **COLLAB**：5,000 個圖、3 個類別（科學合作）
- **NCI1**：4,110 個圖、2 個類別（化學化合物）
- **DD**：1,178 個圖、2 個類別（蛋白質結構）

```python
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
```

### MoleculeNet
**用途**：分子性質預測
**資料集**：超過 10 個分子基準資料集
**描述**：全面的分子機器學習基準
- **ESOL**：水溶解度（迴歸）
- **FreeSolv**：水合自由能（迴歸）
- **Lipophilicity**：辛醇/水分配（迴歸）
- **BACE**：結合結果（分類）
- **BBBP**：血腦屏障穿透（分類）
- **HIV**：HIV 抑制（分類）
- **Tox21**：毒性預測（多任務分類）
- **ToxCast**：毒理學預測（多任務分類）
- **SIDER**：副作用（多任務分類）
- **ClinTox**：臨床試驗毒性（多任務分類）

```python
from torch_geometric.datasets import MoleculeNet
dataset = MoleculeNet(root='/tmp/ESOL', name='ESOL')
```

## 分子和化學資料集

### QM7b
**用途**：分子性質預測（量子力學）
**描述**：7,211 個分子，最多 7 個重原子
- 性質：原子化能、電子性質

```python
from torch_geometric.datasets import QM7b
dataset = QM7b(root='/tmp/QM7b')
```

### QM9
**用途**：分子性質預測（量子力學）
**描述**：約 130,000 個分子，最多 9 個重原子（C、O、N、F）
- 性質：19 種量子化學性質，包括 HOMO、LUMO、能隙、能量

```python
from torch_geometric.datasets import QM9
dataset = QM9(root='/tmp/QM9')
```

### ZINC
**用途**：分子生成、性質預測
**描述**：約 250,000 個類藥分子圖
- 性質：受限溶解度、分子量

```python
from torch_geometric.datasets import ZINC
dataset = ZINC(root='/tmp/ZINC', subset=True)
```

### AQSOL
**用途**：水溶解度預測
**描述**：約 10,000 個具有溶解度測量值的分子

```python
from torch_geometric.datasets import AQSOL
dataset = AQSOL(root='/tmp/AQSOL')
```

### MD17
**用途**：分子動力學（molecular dynamics）、力場學習（force field learning）
**描述**：小分子的分子動力學軌跡
- 分子：苯、尿嘧啶、萘、阿斯匹林、水楊酸等

```python
from torch_geometric.datasets import MD17
dataset = MD17(root='/tmp/MD17', name='benzene')
```

### PCQM4Mv2
**用途**：大規模分子性質預測
**描述**：來自 PubChem 的 380 萬個分子，用於量子化學
- OGB 大規模挑戰的一部分

```python
from torch_geometric.datasets import PCQM4Mv2
dataset = PCQM4Mv2(root='/tmp/PCQM4Mv2')
```

## 社交網路

### Reddit
**用途**：大規模節點分類
**描述**：2014 年 9 月的 Reddit 貼文
- 232,965 個節點、11,606,919 條邊、41 個類別
- 特徵：貼文內容的 TF-IDF

```python
from torch_geometric.datasets import Reddit
dataset = Reddit(root='/tmp/Reddit')
```

### Reddit2
**用途**：大規模節點分類
**描述**：更新版的 Reddit 資料集，包含更多貼文

```python
from torch_geometric.datasets import Reddit2
dataset = Reddit2(root='/tmp/Reddit2')
```

### Twitch
**用途**：節點分類、社交網路分析
**網路**：DE、EN、ES、FR、PT、RU
**描述**：按語言分類的 Twitch 使用者網路

```python
from torch_geometric.datasets import Twitch
dataset = Twitch(root='/tmp/Twitch', name='DE')
```

### Facebook
**用途**：社交網路分析、節點分類
**描述**：Facebook 頁面對頁面網路

```python
from torch_geometric.datasets import FacebookPagePage
dataset = FacebookPagePage(root='/tmp/Facebook')
```

### GitHub
**用途**：社交網路分析
**描述**：GitHub 開發者網路

```python
from torch_geometric.datasets import GitHub
dataset = GitHub(root='/tmp/GitHub')
```

## 知識圖譜

### Entities
**用途**：連結預測（link prediction）、知識圖譜嵌入（knowledge graph embeddings）
**資料集**：AIFB、MUTAG、BGS、AM
**描述**：具有類型化關係的 RDF 知識圖譜

```python
from torch_geometric.datasets import Entities
dataset = Entities(root='/tmp/AIFB', name='AIFB')
```

### WordNet18
**用途**：語義網路上的連結預測
**描述**：具有 18 種關係的 WordNet 子集
- 40,943 個實體、151,442 個三元組

```python
from torch_geometric.datasets import WordNet18
dataset = WordNet18(root='/tmp/WordNet18')
```

### WordNet18RR
**用途**：連結預測（無逆關係）
**描述**：無逆關係的改進版本

```python
from torch_geometric.datasets import WordNet18RR
dataset = WordNet18RR(root='/tmp/WordNet18RR')
```

### FB15k-237
**用途**：Freebase 上的連結預測
**描述**：具有 237 種關係的 Freebase 子集
- 14,541 個實體、310,116 個三元組

```python
from torch_geometric.datasets import FB15k_237
dataset = FB15k_237(root='/tmp/FB15k')
```

## 異質圖

### OGB_MAG
**用途**：異質圖學習、節點分類
**描述**：具有多種節點/邊類型的 Microsoft Academic Graph
- 節點類型：論文、作者、機構、研究領域
- 超過 100 萬個節點、超過 2100 萬條邊

```python
from torch_geometric.datasets import OGB_MAG
dataset = OGB_MAG(root='/tmp/OGB_MAG')
```

### MovieLens
**用途**：推薦系統、連結預測
**版本**：100K、1M、10M、20M
**描述**：使用者-電影評分網路
- 節點類型：使用者、電影
- 邊類型：評分

```python
from torch_geometric.datasets import MovieLens
dataset = MovieLens(root='/tmp/MovieLens', model_name='100k')
```

### IMDB
**用途**：異質圖學習
**描述**：IMDB 電影網路
- 節點類型：電影、演員、導演

```python
from torch_geometric.datasets import IMDB
dataset = IMDB(root='/tmp/IMDB')
```

### DBLP
**用途**：異質圖學習、節點分類
**描述**：DBLP 書目網路
- 節點類型：作者、論文、術語、會議

```python
from torch_geometric.datasets import DBLP
dataset = DBLP(root='/tmp/DBLP')
```

### LastFM
**用途**：異質推薦
**描述**：LastFM 音樂網路
- 節點類型：使用者、藝術家、標籤

```python
from torch_geometric.datasets import LastFM
dataset = LastFM(root='/tmp/LastFM')
```

## 時序圖

### BitcoinOTC
**用途**：時序連結預測、信任網路
**描述**：隨時間變化的 Bitcoin OTC 信任網路

```python
from torch_geometric.datasets import BitcoinOTC
dataset = BitcoinOTC(root='/tmp/BitcoinOTC')
```

### ICEWS18
**用途**：時序知識圖譜補全
**描述**：整合危機早期預警系統事件

```python
from torch_geometric.datasets import ICEWS18
dataset = ICEWS18(root='/tmp/ICEWS18')
```

### GDELT
**用途**：時序事件預測
**描述**：全球事件、語言和語調資料庫

```python
from torch_geometric.datasets import GDELT
dataset = GDELT(root='/tmp/GDELT')
```

### JODIEDataset
**用途**：動態圖學習
**資料集**：Reddit、Wikipedia、MOOC、LastFM
**描述**：時序互動網路

```python
from torch_geometric.datasets import JODIEDataset
dataset = JODIEDataset(root='/tmp/JODIE', name='Reddit')
```

## 3D 網格和點雲

### ShapeNet
**用途**：3D 形狀分類和分割
**描述**：大規模 3D CAD 模型資料集
- 16,881 個模型，橫跨 16 個類別
- 部件級分割標籤

```python
from torch_geometric.datasets import ShapeNet
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
```

### ModelNet
**用途**：3D 形狀分類
**版本**：ModelNet10、ModelNet40
**描述**：用於 3D 物件分類的 CAD 模型
- ModelNet10：4,899 個模型、10 個類別
- ModelNet40：12,311 個模型、40 個類別

```python
from torch_geometric.datasets import ModelNet
dataset = ModelNet(root='/tmp/ModelNet', name='10')
```

### FAUST
**用途**：3D 形狀匹配、對應
**描述**：用於形狀分析的人體掃描
- 100 個網格，10 人 10 種姿勢

```python
from torch_geometric.datasets import FAUST
dataset = FAUST(root='/tmp/FAUST')
```

### CoMA
**用途**：3D 網格變形
**描述**：面部表情網格
- 20,466 個具有表情的 3D 面部掃描

```python
from torch_geometric.datasets import CoMA
dataset = CoMA(root='/tmp/CoMA')
```

### S3DIS
**用途**：3D 語義分割
**描述**：史丹福大規模 3D 室內空間
- 6 個區域、271 個房間、點雲資料

```python
from torch_geometric.datasets import S3DIS
dataset = S3DIS(root='/tmp/S3DIS', test_area=6)
```

## 圖像和視覺資料集

### MNISTSuperpixels
**用途**：基於圖的圖像分類
**描述**：以超像素圖表示的 MNIST 圖像
- 70,000 個圖（60k 訓練、10k 測試）

```python
from torch_geometric.datasets import MNISTSuperpixels
dataset = MNISTSuperpixels(root='/tmp/MNIST')
```

### Flickr
**用途**：圖像描述、節點分類
**描述**：Flickr 圖像網路
- 89,250 個節點、899,756 條邊

```python
from torch_geometric.datasets import Flickr
dataset = Flickr(root='/tmp/Flickr')
```

### PPI
**用途**：蛋白質-蛋白質交互作用預測
**描述**：多圖蛋白質交互作用網路
- 24 個圖，共 2,373 個節點

```python
from torch_geometric.datasets import PPI
dataset = PPI(root='/tmp/PPI', split='train')
```

## 小型經典圖

### KarateClub
**用途**：社群偵測、視覺化
**描述**：Zachary 空手道俱樂部網路
- 34 個節點、78 條邊、2 個社群

```python
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
```

## Open Graph Benchmark (OGB)

PyG 與 OGB 資料集無縫整合：

### 節點性質預測
- **ogbn-products**：Amazon 產品網路（240 萬個節點）
- **ogbn-proteins**：蛋白質關聯網路（13.2 萬個節點）
- **ogbn-arxiv**：引用網路（16.9 萬個節點）
- **ogbn-papers100M**：大型引用網路（1.11 億個節點）
- **ogbn-mag**：異質學術圖

### 連結性質預測
- **ogbl-ppa**：蛋白質關聯網路
- **ogbl-collab**：協作網路
- **ogbl-ddi**：藥物-藥物交互作用網路
- **ogbl-citation2**：引用網路
- **ogbl-wikikg2**：Wikidata 知識圖譜

### 圖性質預測
- **ogbg-molhiv**：分子 HIV 活性預測
- **ogbg-molpcba**：分子生物測定（多任務）
- **ogbg-ppa**：蛋白質功能預測
- **ogbg-code2**：程式碼抽象語法樹

```python
from torch_geometric.datasets import OGB_MAG, OGB_PPA
# 或
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv')
```

## 合成資料集

### FakeDataset
**用途**：測試、除錯
**描述**：生成隨機圖資料

```python
from torch_geometric.datasets import FakeDataset
dataset = FakeDataset(num_graphs=100, avg_num_nodes=50)
```

### StochasticBlockModelDataset
**用途**：社群偵測基準測試
**描述**：由隨機區塊模型生成的圖

```python
from torch_geometric.datasets import StochasticBlockModelDataset
dataset = StochasticBlockModelDataset(root='/tmp/SBM', num_graphs=1000)
```

### ExplainerDataset
**用途**：測試可解釋性方法
**描述**：具有已知解釋基準真相的合成圖

```python
from torch_geometric.datasets import ExplainerDataset
dataset = ExplainerDataset(num_graphs=1000)
```

## 材料科學

### QM8
**用途**：分子性質預測
**描述**：小分子的電子性質

```python
from torch_geometric.datasets import QM8
dataset = QM8(root='/tmp/QM8')
```

## 生物網路

### PPI（蛋白質-蛋白質交互作用）
已在上方圖像和視覺資料集中列出

### STRING
**用途**：蛋白質交互作用網路
**描述**：已知和預測的蛋白質-蛋白質交互作用

```python
# 可透過外部來源或自訂載入取得
```

## 使用提示

1. **從小型資料集開始**：使用 Cora、KarateClub 或 ENZYMES 進行原型設計
2. **引用網路**：Planetoid 資料集非常適合節點分類
3. **圖分類**：TUDataset 提供多樣化的基準測試
4. **分子**：QM9、ZINC、MoleculeNet 適用於化學應用
5. **大規模**：使用 Reddit、OGB 資料集搭配 NeighborLoader
6. **異質**：OGB_MAG、MovieLens、IMDB 適用於多類型圖
7. **時序**：JODIE、ICEWS 適用於動態圖學習
8. **3D**：ShapeNet、ModelNet、S3DIS 適用於幾何學習

## 常見模式

### 搭配轉換載入
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='/tmp/Cora', name='Cora',
                    transform=NormalizeFeatures())
```

### 訓練/驗證/測試分割
```python
# 對於具有預定義分割的資料集
data = dataset[0]
train_data = data[data.train_mask]
val_data = data[data.val_mask]
test_data = data[data.test_mask]

# 對於圖分類
from torch_geometric.loader import DataLoader
train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]
train_loader = DataLoader(train_dataset, batch_size=32)
```

### 自訂資料載入
```python
from torch_geometric.data import Data, Dataset

class MyCustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # 您的初始化

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # 載入並返回資料物件
        return self.data_list[idx]
```
