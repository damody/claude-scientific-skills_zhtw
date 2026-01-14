# 模型和架構

## 概述

TorchDrug 提供各種基於圖的學習任務的綜合預建模型架構集合。本參考文件列出所有可用模型及其特徵、使用案例和實作細節。

## 圖神經網路

### GCN（圖卷積網路）

**類型：**空間訊息傳遞
**論文：**Semi-Supervised Classification with Graph Convolutional Networks（Kipf & Welling, 2017）

**特徵：**
- 簡單高效的聚合
- 正規化鄰接矩陣卷積
- 對同質圖效果好
- 許多任務的良好基線

**最適合：**
- 初始實驗和基線
- 計算效率重要時
- 具有清晰局部結構的圖

**參數：**
- `input_dim`：節點特徵維度
- `hidden_dims`：隱藏層維度列表
- `edge_input_dim`：邊特徵維度（可選）
- `batch_norm`：應用批次正規化
- `activation`：激活函數（relu、elu 等）
- `dropout`：Dropout 率

**使用案例：**
- 分子性質預測
- 引用網路分類
- 社群網路分析

### GAT（圖注意力網路）

**類型：**基於注意力的訊息傳遞
**論文：**Graph Attention Networks（Veličković et al., 2018）

**特徵：**
- 學習鄰居的注意力權重
- 不同鄰居有不同重要性
- 多頭注意力增強穩健性
- 自然處理不同節點度數

**最適合：**
- 鄰居重要性變化時
- 異質圖
- 可解釋的預測

**參數：**
- `input_dim`、`hidden_dims`：標準維度
- `num_heads`：注意力頭數
- `negative_slope`：LeakyReLU 斜率
- `concat`：連接或平均多頭輸出

**使用案例：**
- 蛋白質-蛋白質交互作用預測
- 關注反應位點的分子生成
- 具有關係重要性的知識圖譜推理

### GIN（圖同構網路）

**類型：**最大表達力訊息傳遞
**論文：**How Powerful are Graph Neural Networks?（Xu et al., 2019）

**特徵：**
- 理論上最具表達力的 GNN 架構
- 單射聚合函數
- 可以區分 GCN 無法區分的圖結構
- 分子任務上通常表現最佳

**最適合：**
- 分子性質預測（最先進）
- 需要結構辨別的任務
- 圖分類

**參數：**
- `input_dim`、`hidden_dims`：標準維度
- `edge_input_dim`：包含邊特徵
- `batch_norm`：通常使用 true
- `readout`：圖池化（"sum"、"mean"、"max"）
- `eps`：可學習或固定的 epsilon

**使用案例：**
- 藥物性質預測（BBBP、HIV 等）
- 分子生成
- 反應預測

### RGCN（關係圖卷積網路）

**類型：**多關係訊息傳遞
**論文：**Modeling Relational Data with Graph Convolutional Networks（Schlichtkrull et al., 2018）

**特徵：**
- 處理多種邊/關係類型
- 關係特定的權重矩陣
- 基底分解提高參數效率
- 對知識圖譜至關重要

**最適合：**
- 知識圖譜推理
- 異質分子圖
- 多關係資料

**參數：**
- `num_relation`：關係類型數量
- `hidden_dims`：層維度
- `num_bases`：基底分解（減少參數）

**使用案例：**
- 知識圖譜補全
- 逆合成（不同鍵類型）
- 蛋白質交互作用網路

### MPNN（訊息傳遞神經網路）

**類型：**通用訊息傳遞框架
**論文：**Neural Message Passing for Quantum Chemistry（Gilmer et al., 2017）

**特徵：**
- 靈活的訊息和更新函數
- 訊息計算中的邊特徵
- 用於節點隱藏狀態的 GRU 更新
- 用於圖表示的 Set2Set 讀出

**最適合：**
- 量子化學預測
- 邊資訊重要的任務
- 節點狀態在多次迭代中演化時

**參數：**
- `input_dim`、`hidden_dim`：特徵維度
- `edge_input_dim`：邊特徵維度
- `num_layer`：訊息傳遞迭代次數
- `num_mlp_layer`：訊息函數中的 MLP 層數

**使用案例：**
- QM9 量子性質預測
- 分子動力學
- 3D 構象感知任務

### SchNet（連續濾波卷積網路）

**類型：**3D 幾何感知卷積
**論文：**SchNet: A continuous-filter convolutional neural network（Schütt et al., 2017）

**特徵：**
- 在 3D 原子座標上操作
- 連續濾波卷積
- 旋轉和平移不變
- 量子化學表現優異

**最適合：**
- 3D 分子結構任務
- 量子性質預測
- 蛋白質結構分析
- 能量和力預測

**參數：**
- `input_dim`：原子特徵
- `hidden_dims`：層維度
- `num_gaussian`：距離的 RBF 基函數
- `cutoff`：交互作用截斷距離

**使用案例：**
- QM9 性質預測
- 分子動力學模擬
- 帶結構的蛋白質-配體結合
- 晶體性質預測

### ChebNet（Chebyshev 譜 CNN）

**類型：**譜卷積
**論文：**Convolutional Neural Networks on Graphs（Defferrard et al., 2016）

**特徵：**
- 譜圖卷積
- Chebyshev 多項式近似
- 捕捉全局圖結構
- 計算效率高

**最適合：**
- 需要全局資訊的任務
- 圖拉普拉斯資訊豐富時
- 理論分析

**參數：**
- `input_dim`、`hidden_dims`：維度
- `num_cheb`：Chebyshev 多項式階數

**使用案例：**
- 引用網路分類
- 腦網路分析
- 圖上的訊號處理

### NFP（神經指紋）

**類型：**分子指紋學習
**論文：**Convolutional Networks on Graphs for Learning Molecular Fingerprints（Duvenaud et al., 2015）

**特徵：**
- 學習可微分的分子指紋
- 手工指紋（ECFP）的替代方案
- 類似 ECFP 的圓形卷積
- 可解釋的學習特徵

**最適合：**
- 分子相似性學習
- 有限資料的性質預測
- 可解釋性重要時

**參數：**
- `input_dim`、`output_dim`：特徵維度
- `hidden_dims`：層維度
- `num_layer`：圓形卷積深度

**使用案例：**
- 虛擬篩選
- 分子相似性搜尋
- QSAR 建模

## 蛋白質專用模型

### GearNet（幾何感知關係圖網路）

**類型：**蛋白質結構編碼器
**論文：**Protein Representation Learning by Geometric Structure Pretraining（Zhang et al., 2023）

**特徵：**
- 整合 3D 幾何資訊
- 多種邊類型（序列、空間、KNN）
- 專為蛋白質設計
- 蛋白質任務達到最先進水準

**最適合：**
- 蛋白質結構預測
- 蛋白質功能預測
- 蛋白質-蛋白質交互作用
- 任何有蛋白質 3D 結構的任務

**參數：**
- `input_dim`：殘基特徵
- `hidden_dims`：層維度
- `num_relation`：邊類型（序列、半徑、KNN）
- `edge_input_dim`：幾何特徵（距離、角度）
- `batch_norm`：通常使用 true

**使用案例：**
- 酵素功能預測（EnzymeCommission）
- 蛋白質摺疊識別
- 接觸預測
- 結合位點識別

### ESM（演化規模建模）

**類型：**蛋白質語言模型（transformer）
**論文：**Biological structure and function emerge from scaling unsupervised learning（Rives et al., 2021）

**特徵：**
- 在 2.5 億+ 蛋白質序列上預訓練
- 捕捉演化和結構資訊
- Transformer 架構
- 下游任務的遷移學習

**最適合：**
- 任何基於序列的蛋白質任務
- 沒有結構可用時
- 有限資料的遷移學習

**變體：**
- ESM-1b：6.5 億參數
- ESM-2：多種大小（8M 到 15B 參數）

**使用案例：**
- 蛋白質功能預測
- 變異效果預測
- 蛋白質設計
- 結構預測（ESMFold）

### ProteinBERT

**類型：**蛋白質的遮罩語言模型

**特徵：**
- BERT 風格預訓練
- 遮罩胺基酸預測
- 雙向上下文
- 適合基於序列的任務

**使用案例：**
- 功能註釋
- 亞細胞定位
- 穩定性預測

### ProteinCNN / ProteinResNet

**類型：**序列的卷積網路

**特徵：**
- 序列上的 1D 卷積
- 局部模式識別
- 比 transformer 更快
- 適合基序檢測

**使用案例：**
- 結合位點預測
- 二級結構預測
- 結構域識別

### ProteinLSTM

**類型：**序列的循環網路

**特徵：**
- 雙向 LSTM
- 捕捉長距離依賴性
- 序列處理
- 序列任務的良好基線

**使用案例：**
- 順序預測
- 序列註釋
- 時間序列蛋白質資料

## 知識圖譜模型

### TransE（平移嵌入）

**類型：**基於平移的嵌入
**論文：**Translating Embeddings for Modeling Multi-relational Data（Bordes et al., 2013）

**特徵：**
- h + r ≈ t（頭 + 關係 ≈ 尾）
- 簡單且可解釋
- 對 1 對 1 關係效果好
- 記憶體效率高

**最適合：**
- 大型知識圖譜
- 初始實驗
- 可解釋嵌入

**參數：**
- `num_entity`、`num_relation`：圖大小
- `embedding_dim`：嵌入維度（通常 50-500）

### RotatE（旋轉嵌入）

**類型：**複數空間中的旋轉
**論文：**RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space（Sun et al., 2019）

**特徵：**
- 關係作為複數空間中的旋轉
- 處理對稱、反對稱、逆向、組合
- 在許多基準上達到最先進水準

**最適合：**
- 大多數知識圖譜任務
- 複雜關係模式
- 準確度關鍵時

**參數：**
- `num_entity`、`num_relation`：圖大小
- `embedding_dim`：必須為偶數（複數嵌入）
- `max_score`：分數裁剪值

### DistMult

**類型：**雙線性模型

**特徵：**
- 對稱關係建模
- 快速高效
- 無法建模非對稱關係

**最適合：**
- 對稱關係（如「相似於」）
- 速度關鍵時
- 大規模圖

### ComplEx

**類型：**複數值嵌入

**特徵：**
- 處理非對稱和對稱關係
- 對大多數圖比 DistMult 更好
- 表達能力和效率的良好平衡

**最適合：**
- 通用知識圖譜補全
- 混合關係類型
- 當 RotatE 太複雜時

### SimplE

**類型：**增強嵌入模型

**特徵：**
- 每個實體有兩個嵌入（規範 + 逆向）
- 完全表達性
- 比 ComplEx 參數略多

**最適合：**
- 需要完全表達性時
- 逆向關係重要時

## 生成模型

### GraphAutoregressiveFlow

**類型：**分子的正規化流

**特徵：**
- 精確似然計算
- 可逆變換
- 穩定訓練（無對抗）
- 支援條件生成

**最適合：**
- 分子生成
- 密度估計
- 分子間插值

**參數：**
- `input_dim`：原子特徵
- `hidden_dims`：耦合層
- `num_flow`：流變換數量

**使用案例：**
- 從頭藥物設計
- 化學空間探索
- 性質目標生成

## 預訓練模型

### InfoGraph

**類型：**對比學習

**特徵：**
- 最大化互資訊
- 圖級和節點級對比
- 無監督預訓練
- 對小型資料集效果好

**使用案例：**
- 預訓練分子編碼器
- 少樣本學習
- 遷移學習

### MultiviewContrast

**類型：**蛋白質的多視角對比學習

**特徵：**
- 對比蛋白質的不同視角
- 幾何預訓練
- 使用 3D 結構資訊
- 對蛋白質模型表現優異

**使用案例：**
- 在蛋白質結構上預訓練 GearNet
- 遷移到性質預測
- 有限標註資料場景

## 模型選擇指南

### 按任務類型

**分子性質預測：**
1. GIN（首選）
2. GAT（可解釋性）
3. SchNet（3D 可用）

**蛋白質任務：**
1. ESM（僅序列）
2. GearNet（結構可用）
3. ProteinBERT（序列，比 ESM 輕量）

**知識圖譜：**
1. RotatE（最佳效能）
2. ComplEx（良好平衡）
3. TransE（大型圖、效率）

**分子生成：**
1. GraphAutoregressiveFlow（精確似然）
2. 帶 GIN 骨幹的 GCPN（性質最佳化）

**逆合成：**
1. GIN（合成子完成）
2. RGCN（帶鍵類型的中心識別）

### 按資料集大小

**小型（< 1k）：**
- 使用預訓練模型（蛋白質用 ESM）
- 更簡單的架構（GCN、ProteinCNN）
- 強正則化

**中型（1k-100k）：**
- 分子用 GIN
- 可解釋性用 GAT
- 標準訓練

**大型（> 100k）：**
- 任何模型都可以
- 更深的架構
- 可以從頭訓練

### 按計算預算

**低：**
- GCN（最簡單）
- DistMult（KG）
- ProteinLSTM

**中：**
- GIN
- GAT
- ComplEx

**高：**
- ESM（大型）
- SchNet（3D）
- 高維 RotatE

## 實作技巧

1. **從簡單開始**：以 GCN 或 GIN 作為基線
2. **使用預訓練**：蛋白質用 ESM，分子用 InfoGraph
3. **調整深度**：通常 3-5 層就足夠
4. **批次正規化**：通常有幫助（除了 KG 嵌入）
5. **殘差連接**：對深層網路重要
6. **讀出函數**：「mean」通常效果好
7. **邊特徵**：可用時包含（鍵、距離）
8. **正則化**：Dropout、權重衰減、早停
