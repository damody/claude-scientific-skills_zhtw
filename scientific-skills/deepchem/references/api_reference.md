# DeepChem API 參考

本文件提供按功能組織的 DeepChem 核心 API 全面參考。

## 資料處理

### 資料載入器

#### 檔案格式載入器
- **CSVLoader**：從 CSV 檔案載入表格資料，具有可自訂的特徵處理
- **UserCSVLoader**：使用者定義的 CSV 載入，具有靈活的欄位規格
- **SDFLoader**：處理分子結構檔案（SDF 格式）
- **JsonLoader**：匯入 JSON 結構化資料集
- **ImageLoader**：載入電腦視覺任務的影像資料

#### 生物資料載入器
- **FASTALoader**：處理 FASTA 格式的蛋白質/DNA 序列
- **FASTQLoader**：處理帶有品質分數的 FASTQ 定序資料
- **SAMLoader/BAMLoader/CRAMLoader**：支援序列比對格式

#### 專門載入器
- **DFTYamlLoader**：處理密度泛函理論計算資料
- **InMemoryLoader**：直接從 Python 物件載入資料

### 資料集類別

- **NumpyDataset**：包裝 NumPy 陣列用於記憶體內資料操作
- **DiskDataset**：管理儲存在磁碟上的大型資料集，減少記憶體開銷
- **ImageDataset**：用於影像相關機器學習任務的專門容器

### 資料分割器

#### 通用分割器
- **RandomSplitter**：隨機資料集分割
- **IndexSplitter**：按指定索引分割
- **SpecifiedSplitter**：使用預定義分割
- **RandomStratifiedSplitter**：分層隨機分割
- **SingletaskStratifiedSplitter**：單一任務的分層分割
- **TaskSplitter**：多任務情境的分割

#### 分子專用分割器
- **ScaffoldSplitter**：按結構骨架劃分分子（防止資料洩漏）
- **ButinaSplitter**：基於聚類的分子分割
- **FingerprintSplitter**：基於分子指紋相似性分割
- **MaxMinSplitter**：最大化訓練/測試集之間的多樣性
- **MolecularWeightSplitter**：按分子量屬性分割

**最佳實踐**：對於藥物發現任務，使用 ScaffoldSplitter 以防止對相似分子結構的過擬合。

### 轉換器

#### 正規化
- **NormalizationTransformer**：標準正規化（mean=0, std=1）
- **MinMaxTransformer**：將特徵縮放到 [0,1] 範圍
- **LogTransformer**：應用對數轉換
- **PowerTransformer**：Box-Cox 和 Yeo-Johnson 轉換
- **CDFTransformer**：累積分佈函數正規化

#### 任務專用
- **BalancingTransformer**：解決類別不平衡問題
- **FeaturizationTransformer**：應用動態特徵工程
- **CoulombFitTransformer**：量子化學專用
- **DAGTransformer**：有向無環圖轉換
- **RxnSplitTransformer**：化學反應預處理

## 分子特徵化器

### 基於圖的特徵化器
與圖神經網路（GCN、MPNN 等）一起使用：

- **ConvMolFeaturizer**：圖卷積網路的圖表示
- **WeaveFeaturizer**：「Weave」圖嵌入
- **MolGraphConvFeaturizer**：圖卷積就緒的表示
- **EquivariantGraphFeaturizer**：保持幾何不變性
- **DMPNNFeaturizer**：有向訊息傳遞神經網路輸入
- **GroverFeaturizer**：預訓練分子嵌入

### 基於指紋的特徵化器
與傳統機器學習（Random Forest、SVM、XGBoost）一起使用：

- **MACCSKeysFingerprint**：167 位元結構鍵
- **CircularFingerprint**：擴展連接指紋（Morgan 指紋）
  - 參數：`radius`（預設 2）、`size`（預設 2048）、`useChirality`（預設 False）
- **PubChemFingerprint**：881 位元結構描述子
- **Mol2VecFingerprint**：學習的分子向量表示

### 描述子特徵化器
直接計算分子屬性：

- **RDKitDescriptors**：約 200 個分子描述子（MW、LogP、H-供體、H-受體、TPSA 等）
- **MordredDescriptors**：全面的結構和物理化學描述子
- **CoulombMatrix**：3D 結構的原子間距離矩陣

### 基於序列的特徵化器
用於循環網路和 Transformer：

- **SmilesToSeq**：將 SMILES 字串轉換為序列
- **SmilesToImage**：從 SMILES 生成 2D 影像表示
- **RawFeaturizer**：不變地傳遞原始分子資料

### 選擇指南

| 使用案例 | 建議特徵化器 | 模型類型 |
|----------|------------|---------|
| 圖神經網路 | ConvMolFeaturizer、MolGraphConvFeaturizer | GCN、MPNN、GAT |
| 傳統機器學習 | CircularFingerprint、RDKitDescriptors | Random Forest、XGBoost、SVM |
| 深度學習（非圖） | CircularFingerprint、Mol2VecFingerprint | 密集網路、CNN |
| 序列模型 | SmilesToSeq | LSTM、GRU、Transformer |
| 3D 分子結構 | CoulombMatrix | 專門 3D 模型 |
| 快速基線 | RDKitDescriptors | Linear、Ridge、Lasso |

## 模型

### Scikit-Learn 整合
- **SklearnModel**：任何 scikit-learn 演算法的包裝器
  - 用法：`SklearnModel(model=RandomForestRegressor())`

### 梯度提升
- **GBDTModel**：梯度提升決策樹（XGBoost、LightGBM）

### PyTorch 模型

#### 分子屬性預測
- **MultitaskRegressor**：具有共享表示的多任務迴歸
- **MultitaskClassifier**：多任務分類
- **MultitaskFitTransformRegressor**：具有學習轉換的迴歸
- **GCNModel**：圖卷積網路
- **GATModel**：圖注意力網路
- **AttentiveFPModel**：注意力指紋網路
- **DMPNNModel**：有向訊息傳遞神經網路
- **GroverModel**：GROVER 預訓練 Transformer
- **MATModel**：分子注意力 Transformer

#### 材料科學
- **CGCNNModel**：晶體圖卷積網路
- **MEGNetModel**：材料圖網路
- **LCNNModel**：用於材料的晶格 CNN

#### 生成式模型
- **GANModel**：生成對抗網路
- **WGANModel**：Wasserstein GAN
- **BasicMolGANModel**：分子 GAN
- **LSTMGenerator**：基於 LSTM 的分子生成
- **SeqToSeqModel**：序列到序列模型

#### 物理資訊模型
- **PINNModel**：物理資訊神經網路
- **HNNModel**：哈密頓神經網路
- **LNN**：拉格朗日神經網路
- **FNOModel**：傅立葉神經運算子

#### 電腦視覺
- **CNN**：卷積神經網路
- **UNetModel**：用於分割的 U-Net 架構
- **InceptionV3Model**：預訓練 Inception v3
- **MobileNetV2Model**：輕量級行動網路

### Hugging Face 模型

- **HuggingFaceModel**：HF Transformer 的通用包裝器
- **Chemberta**：用於分子屬性預測的化學 BERT
- **MoLFormer**：分子 Transformer 架構
- **ProtBERT**：蛋白質序列 BERT
- **DeepAbLLM**：抗體大型語言模型

### 模型選擇指南

| 任務 | 建議模型 | 特徵化器 |
|------|---------|---------|
| 小型資料集（<1000 樣本） | SklearnModel（Random Forest） | CircularFingerprint |
| 中型資料集（1K-100K） | GBDTModel 或 MultitaskRegressor | CircularFingerprint 或 ConvMolFeaturizer |
| 大型資料集（>100K） | GCNModel、AttentiveFPModel 或 DMPNN | MolGraphConvFeaturizer |
| 遷移學習 | GroverModel、Chemberta、MoLFormer | 模型專用 |
| 材料屬性 | CGCNNModel、MEGNetModel | 基於結構 |
| 分子生成 | BasicMolGANModel、LSTMGenerator | SmilesToSeq |
| 蛋白質序列 | ProtBERT | 基於序列 |

## MoleculeNet 資料集

通過 `dc.molnet.load_*()` 函數快速存取 30+ 個基準資料集。

### 分類資料集
- **load_bace()**：BACE-1 抑制劑（二元分類）
- **load_bbbp()**：血腦屏障穿透
- **load_clintox()**：臨床毒性
- **load_hiv()**：HIV 抑制活性
- **load_muv()**：PubChem BioAssay（具挑戰性，稀疏）
- **load_pcba()**：PubChem 篩選資料
- **load_sider()**：藥物不良反應（多標籤）
- **load_tox21()**：12 種毒性測定（多任務）
- **load_toxcast()**：EPA ToxCast 篩選

### 迴歸資料集
- **load_delaney()**：水溶性（ESOL）
- **load_freesolv()**：溶劑化自由能
- **load_lipo()**：親脂性（辛醇-水分配）
- **load_qm7/qm8/qm9()**：量子力學屬性
- **load_hopv()**：有機光伏屬性

### 蛋白質-配體結合
- **load_pdbbind()**：結合親和力資料

### 材料科學
- **load_perovskite()**：鈣鈦礦穩定性
- **load_mp_formation_energy()**：Materials Project 形成能
- **load_mp_metallicity()**：金屬 vs. 非金屬分類
- **load_bandgap()**：電子能隙預測

### 化學反應
- **load_uspto()**：USPTO 反應資料集

### 使用模式
```python
tasks, datasets, transformers = dc.molnet.load_bbbp(
    featurizer='GraphConv',  # 或 'ECFP'、'GraphConv'、'Weave' 等
    splitter='scaffold',      # 或 'random'、'stratified' 等
    reload=False              # 設為 True 以跳過快取
)
train, valid, test = datasets
```

## 指標

`dc.metrics` 中可用的常見評估指標：

### 分類指標
- **roc_auc_score**：ROC 曲線下面積（二元/多類別）
- **prc_auc_score**：精確度-召回率曲線下面積
- **accuracy_score**：分類準確度
- **balanced_accuracy_score**：不平衡資料集的平衡準確度
- **recall_score**：靈敏度/召回率
- **precision_score**：精確度
- **f1_score**：F1 分數

### 迴歸指標
- **mean_absolute_error**：MAE
- **mean_squared_error**：MSE
- **root_mean_squared_error**：RMSE
- **r2_score**：R² 決定係數
- **pearson_r2_score**：皮爾森相關
- **spearman_correlation**：斯皮爾曼等級相關

### 多任務指標
大多數指標通過跨任務平均支援多任務評估。

## 訓練模式

標準 DeepChem 工作流程：

```python
# 1. 載入資料
loader = dc.data.CSVLoader(tasks=['task1'], feature_field='smiles',
                           featurizer=dc.feat.CircularFingerprint())
dataset = loader.create_dataset('data.csv')

# 2. 分割資料
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

# 3. 轉換資料（可選）
transformers = [dc.trans.NormalizationTransformer(dataset=train)]
for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

# 4. 建立並訓練模型
model = dc.models.MultitaskRegressor(n_tasks=1, n_features=2048, layer_sizes=[1000])
model.fit(train, nb_epoch=50)

# 5. 評估
metric = dc.metrics.Metric(dc.metrics.r2_score)
train_score = model.evaluate(train, [metric])
test_score = model.evaluate(test, [metric])
```

## 常見模式

### 模式 1：使用 MoleculeNet 的快速基線
```python
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
train, valid, test = datasets
model = dc.models.MultitaskClassifier(n_tasks=len(tasks), n_features=1024)
model.fit(train)
```

### 模式 2：使用圖網路的自訂資料
```python
featurizer = dc.feat.MolGraphConvFeaturizer()
loader = dc.data.CSVLoader(tasks=['activity'], feature_field='smiles',
                           featurizer=featurizer)
dataset = loader.create_dataset('my_data.csv')
train, test = dc.splits.RandomSplitter().train_test_split(dataset)
model = dc.models.GCNModel(mode='classification', n_tasks=1)
model.fit(train)
```

### 模式 3：使用預訓練模型的遷移學習
```python
model = dc.models.GroverModel(task='classification', n_tasks=1)
model.fit(train_dataset)
predictions = model.predict(test_dataset)
```
