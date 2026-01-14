# PyHealth 模型

## 概述

PyHealth 提供 33+ 個用於醫療預測任務的模型，從簡單的基準模型到最先進的深度學習架構。模型分為通用架構和醫療專用模型。

## 模型基礎類別

所有模型繼承自 `BaseModel`，具有標準 PyTorch 功能：

**關鍵屬性：**
- `dataset`：關聯的 SampleDataset
- `feature_keys`：要使用的輸入特徵（例如 ["diagnoses", "medications"]）
- `mode`：任務類型（"binary"、"multiclass"、"multilabel"、"regression"）
- `embedding_dim`：特徵嵌入維度
- `device`：運算裝置（CPU/GPU）

**關鍵方法：**
- `forward()`：模型前向傳遞
- `train_step()`：單次訓練迭代
- `eval_step()`：單次評估迭代
- `save()`：儲存模型檢查點
- `load()`：載入模型檢查點

## 通用模型

### 基準模型

**Logistic Regression**（`LogisticRegression`）
- 使用平均池化的線性分類器
- 用於比較的簡單基準
- 快速訓練與推論
- 適合可解釋性需求

**使用方式：**
```python
from pyhealth.models import LogisticRegression

model = LogisticRegression(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary"
)
```

**Multi-Layer Perceptron**（`MLP`）
- 前饋神經網路
- 可配置隱藏層
- 支援 mean/sum/max 池化
- 結構化資料的良好基準

**參數：**
- `hidden_dim`：隱藏層大小
- `num_layers`：隱藏層數量
- `dropout`：Dropout 率
- `pooling`：聚合方法（"mean"、"sum"、"max"）

**使用方式：**
```python
from pyhealth.models import MLP

model = MLP(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    hidden_dim=128,
    num_layers=3,
    dropout=0.5
)
```

### 卷積神經網路

**CNN**（`CNN`）
- 用於模式檢測的卷積層
- 對序列和空間資料有效
- 捕捉局部時間模式
- 參數效率高

**架構：**
- 多個一維卷積層
- 最大池化進行維度縮減
- 全連接輸出層

**參數：**
- `num_filters`：卷積濾波器數量
- `kernel_size`：卷積核大小
- `num_layers`：卷積層數量
- `dropout`：Dropout 率

**使用方式：**
```python
from pyhealth.models import CNN

model = CNN(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    num_filters=64,
    kernel_size=3,
    num_layers=3
)
```

**Temporal Convolutional Networks**（`TCN`）
- 用於長距離依賴的擴張卷積
- 因果卷積（無未來資訊洩漏）
- 對長序列效率高
- 適合時間序列預測

**優勢：**
- 捕捉長期依賴
- 可平行化（比 RNN 快）
- 梯度穩定

### 循環神經網路

**RNN**（`RNN`）
- 基本循環架構
- 支援 LSTM、GRU、RNN 變體
- 序列處理
- 捕捉時間依賴

**參數：**
- `rnn_type`："LSTM"、"GRU" 或 "RNN"
- `hidden_dim`：隱藏狀態維度
- `num_layers`：循環層數量
- `dropout`：Dropout 率
- `bidirectional`：使用雙向 RNN

**使用方式：**
```python
from pyhealth.models import RNN

model = RNN(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    rnn_type="LSTM",
    hidden_dim=128,
    num_layers=2,
    bidirectional=True
)
```

**最適合：**
- 序列臨床事件
- 時間模式學習
- 變長序列

### Transformer 模型

**Transformer**（`Transformer`）
- 自注意力機制
- 序列平行處理
- 最先進效能
- 對長距離依賴有效

**架構：**
- 多頭自注意力
- 位置嵌入
- 前饋網路
- 層標準化

**參數：**
- `num_heads`：注意力頭數量
- `num_layers`：Transformer 層數量
- `hidden_dim`：隱藏維度
- `dropout`：Dropout 率
- `max_seq_length`：最大序列長度

**使用方式：**
```python
from pyhealth.models import Transformer

model = Transformer(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    num_heads=8,
    num_layers=6,
    hidden_dim=256,
    dropout=0.1
)
```

**TransformersModel**（`TransformersModel`）
- 與 HuggingFace transformers 整合
- 用於臨床文字的預訓練語言模型
- 針對醫療任務的微調
- 範例：BERT、RoBERTa、BioClinicalBERT

**使用方式：**
```python
from pyhealth.models import TransformersModel

model = TransformersModel(
    dataset=sample_dataset,
    feature_keys=["text"],
    mode="multiclass",
    pretrained_model="emilyalsentzer/Bio_ClinicalBERT"
)
```

### 圖神經網路

**GNN**（`GNN`）
- 基於圖的學習
- 建模實體間關係
- 支援 GAT（圖注意力）和 GCN（圖卷積）

**使用情境：**
- 藥物交互作用
- 病人相似性網路
- 知識圖譜整合
- 共病關係

**參數：**
- `gnn_type`："GAT" 或 "GCN"
- `hidden_dim`：隱藏維度
- `num_layers`：GNN 層數量
- `dropout`：Dropout 率
- `num_heads`：注意力頭數（用於 GAT）

**使用方式：**
```python
from pyhealth.models import GNN

model = GNN(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="multilabel",
    gnn_type="GAT",
    hidden_dim=128,
    num_layers=3,
    num_heads=4
)
```

## 醫療專用模型

### 可解釋臨床模型

**RETAIN**（`RETAIN`）
- 反向時間注意力機制
- 高度可解釋的預測
- 就診層級和事件層級注意力
- 識別有影響力的臨床事件

**關鍵特徵：**
- 兩層注意力（就診和特徵）
- 時間衰減建模
- 臨床意義的解釋
- 發表於 NeurIPS 2016

**使用方式：**
```python
from pyhealth.models import RETAIN

model = RETAIN(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    hidden_dim=128
)

# 取得注意力權重進行解釋
outputs = model(batch)
visit_attention = outputs["visit_attention"]
feature_attention = outputs["feature_attention"]
```

**最適合：**
- 死亡率預測
- 再入院預測
- 臨床風險評分
- 可解釋預測

**AdaCare**（`AdaCare`）
- 具特徵校準的自適應照護模型
- 疾病特定注意力
- 處理不規則時間間隔
- 可解釋的特徵重要性

**ConCare**（`ConCare`）
- 跨就診卷積注意力
- 時間卷積特徵提取
- 多層注意力機制
- 適合縱向 EHR 建模

### 藥物推薦模型

**GAMENet**（`GAMENet`）
- 基於圖的藥物推薦
- 藥物交互作用建模
- 用於病人歷史的記憶網路
- 多跳推理

**架構：**
- 藥物知識圖譜
- 記憶增強神經網路
- DDI 感知預測

**使用方式：**
```python
from pyhealth.models import GAMENet

model = GAMENet(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="multilabel",
    embedding_dim=128,
    ddi_adj_path="/path/to/ddi_adjacency_matrix.pkl"
)
```

**MICRON**（`MICRON`）
- 帶 DDI 約束的藥物推薦
- 交互作用感知預測
- 安全導向的藥物選擇

**SafeDrug**（`SafeDrug`）
- 安全感知藥物推薦
- 分子結構整合
- DDI 約束優化
- 平衡療效與安全性

**關鍵特徵：**
- 分子圖編碼
- DDI 圖神經網路
- 用於安全的強化學習
- 發表於 KDD 2021

**使用方式：**
```python
from pyhealth.models import SafeDrug

model = SafeDrug(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="multilabel",
    ddi_adj_path="/path/to/ddi_matrix.pkl",
    molecule_path="/path/to/molecule_graphs.pkl"
)
```

**MoleRec**（`MoleRec`）
- 分子層級藥物推薦
- 子結構推理
- 細粒度藥物選擇

### 疾病進展模型

**StageNet**（`StageNet`）
- 疾病階段感知預測
- 自動學習臨床階段
- 階段自適應特徵提取
- 對慢性病監測有效

**架構：**
- 階段感知 LSTM
- 動態階段轉換
- 時間衰減機制

**使用方式：**
```python
from pyhealth.models import StageNet

model = StageNet(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",
    hidden_dim=128,
    num_stages=3,
    chunk_size=128
)
```

**最適合：**
- 加護病房死亡率預測
- 慢性病進展
- 時變風險評估

**Deepr**（`Deepr`）
- 深度循環架構
- 醫療概念嵌入
- 時間模式學習
- 發表於 JAMIA

### 進階序列模型

**Agent**（`Agent`）
- 基於強化學習
- 治療推薦
- 動作價值優化
- 序列決策的策略學習

**GRASP**（`GRASP`）
- 基於圖的序列模式
- 結構事件關係
- 階層表示學習

**SparcNet**（`SparcNet`）
- 稀疏臨床網路
- 高效特徵選擇
- 降低運算成本
- 可解釋預測

**ContraWR**（`ContraWR`）
- 對比學習方法
- 自監督預訓練
- 魯棒表示
- 有限標註資料場景

### 醫療實體連結

**MedLink**（`MedLink`）
- 醫療實體連結到知識庫
- 臨床概念標準化
- UMLS 整合
- 實體消歧

### 生成模型

**GAN**（`GAN`）
- 生成對抗網路
- 合成 EHR 資料生成
- 隱私保護資料共享
- 罕見疾病資料增強

**VAE**（`VAE`）
- 變分自編碼器
- 病人表示學習
- 異常檢測
- 潛在空間探索

### 健康社會決定因素

**SDOH**（`SDOH`）
- 社會決定因素整合
- 多模態預測
- 解決健康不平等
- 結合臨床與社會資料

## 模型選擇指南

### 按任務類型

**二元分類**（死亡率、再入院）
- 起始：Logistic Regression（基準）
- 標準：RNN、Transformer
- 可解釋：RETAIN、AdaCare
- 進階：StageNet

**多標籤分類**（藥物推薦）
- 標準：CNN、RNN
- 醫療專用：GAMENet、SafeDrug、MICRON、MoleRec
- 基於圖：GNN

**迴歸**（住院天數）
- 起始：MLP（基準）
- 序列：RNN、TCN
- 進階：Transformer

**多類別分類**（醫療編碼、專科）
- 標準：CNN、RNN、Transformer
- 文字：TransformersModel（BERT 變體）

### 按資料類型

**序列事件**（診斷、藥物、處置）
- RNN、LSTM、GRU
- Transformer
- RETAIN、AdaCare、ConCare

**時間序列訊號**（EEG、ECG）
- CNN、TCN
- RNN
- Transformer

**文字**（臨床筆記）
- TransformersModel（ClinicalBERT、BioBERT）
- CNN 用於較短文字
- RNN 用於序列文字

**圖**（藥物交互作用、病人網路）
- GNN（GAT、GCN）
- GAMENet、SafeDrug

**影像**（X 光、CT 掃描）
- CNN（透過 TransformersModel 使用 ResNet、DenseNet）
- Vision Transformers

### 按可解釋性需求

**需要高可解釋性：**
- Logistic Regression
- RETAIN
- AdaCare
- SparcNet

**中等可解釋性：**
- CNN（濾波器視覺化）
- Transformer（注意力視覺化）
- GNN（圖注意力）

**可接受黑盒：**
- 深度 RNN 模型
- 複雜集成模型

## 訓練考量

### 超參數調整

**嵌入維度：**
- 小型資料集：64-128
- 大型資料集：128-256
- 複雜任務：256-512

**隱藏維度：**
- 與 embedding_dim 成比例
- 通常為 embedding_dim 的 1-2 倍

**層數：**
- 從 2-3 層開始
- 複雜模式用更深
- 注意過擬合

**Dropout：**
- 從 0.5 開始
- 欠擬合時降低（0.1-0.3）
- 過擬合時增加（0.5-0.7）

### 運算需求

**記憶體（GPU）：**
- CNN：低到中等
- RNN：中等（取決於序列長度）
- Transformer：高（與序列長度呈二次關係）
- GNN：中到高（取決於圖大小）

**訓練速度：**
- 最快：Logistic Regression、MLP、CNN
- 中等：RNN、GNN
- 較慢：Transformer（但可平行化）

### 最佳實務

1. **從簡單基準開始**（Logistic Regression、MLP）
2. **根據資料可用性使用適當的特徵鍵**
3. **將模式與任務輸出匹配**（binary、multiclass、multilabel、regression）
4. **考慮臨床部署的可解釋性需求**
5. **在保留測試集上驗證**以獲得實際效能
6. **監控過擬合**尤其是複雜模型
7. 盡可能**使用預訓練模型**（TransformersModel）
8. **考慮部署的運算限制**

## 範例工作流程

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer

# 1. 準備資料
dataset = MIMIC4Dataset(root="/path/to/data")
sample_dataset = dataset.set_task(mortality_prediction_mimic4_fn)

# 2. 初始化模型
model = Transformer(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications", "procedures"],
    mode="binary",
    embedding_dim=128,
    num_heads=8,
    num_layers=3,
    dropout=0.3
)

# 3. 訓練模型
trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=50,
    monitor="pr_auc_score",
    monitor_criterion="max"
)

# 4. 評估
results = trainer.evaluate(test_loader)
print(results)
```
