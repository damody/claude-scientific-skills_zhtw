# Molfeat 可用的特徵提取器

本文件提供 molfeat 中所有可用特徵提取器的完整目錄，按類別組織。

## 基於 Transformer 的語言模型

使用 SMILES/SELFIES 表示的預訓練 transformer 分子嵌入模型。

### RoBERTa 風格模型
- **Roberta-Zinc480M-102M** - 在 ZINC 資料庫約 4.8 億個 SMILES 字串上訓練的 RoBERTa 掩碼語言模型
- **ChemBERTa-77M-MLM** - 基於 RoBERTa、在 7700 萬個 PubChem 化合物上訓練的掩碼語言模型
- **ChemBERTa-77M-MTR** - 在 PubChem 化合物上訓練的多任務迴歸版本

### GPT 風格自迴歸模型
- **GPT2-Zinc480M-87M** - 在 ZINC 約 4.8 億個 SMILES 上訓練的 GPT-2 自迴歸語言模型
- **ChemGPT-1.2B** - 在 PubChem10M 上預訓練的大型 transformer（12 億參數）
- **ChemGPT-19M** - 在 PubChem10M 上預訓練的中型 transformer（1900 萬參數）
- **ChemGPT-4.7M** - 在 PubChem10M 上預訓練的小型 transformer（470 萬參數）

### 專門 Transformer 模型
- **MolT5** - 用於分子描述和基於文字生成的自監督框架

## 圖神經網路（GNNs）

在分子圖結構上運作的預訓練圖神經網路模型。

### GIN（圖同構網路）變體
全部在 ChEMBL 分子上以不同目標預訓練：
- **gin-supervised-masking** - 帶節點掩碼目標的監督式
- **gin-supervised-infomax** - 帶圖級互資訊最大化的監督式
- **gin-supervised-edgepred** - 帶邊緣預測目標的監督式
- **gin-supervised-contextpred** - 帶上下文預測目標的監督式

### 其他圖形模型
- **JTVAE_zinc_no_kl** - 用於分子生成的接點樹 VAE（在 ZINC 上訓練）
- **Graphormer-pcqm4mv2** - 在 PCQM4Mv2 量子化學資料集上預訓練的圖 transformer，用於 HOMO-LUMO 能隙預測

## 分子描述符

用於物理化學性質和分子特徵的計算器。

### 2D 描述符
- **desc2D** / **rdkit2D** - 200+ RDKit 2D 分子描述符，包括：
  - 分子量、logP、TPSA
  - 氫鍵供體/受體
  - 可旋轉鍵
  - 環計數和芳香性
  - 分子複雜度指標

### 3D 描述符
- **desc3D** / **rdkit3D** - RDKit 3D 分子描述符（需要構象生成）
  - 慣性矩
  - PMI（主慣性矩）比率
  - 非球形性、偏心率
  - 迴轉半徑

### 全面描述符集
- **mordred** - 超過 1800 個分子描述符，涵蓋：
  - 構成描述符
  - 拓撲指數
  - 連接性指數
  - 資訊含量
  - 2D/3D 自相關
  - WHIM 描述符
  - GETAWAY 描述符
  - 以及更多

### 電拓撲描述符
- **estate** - 電拓撲狀態（E-State）指數，編碼：
  - 原子環境資訊
  - 電子和拓撲性質
  - 雜原子貢獻

## 分子指紋

表示分子子結構的二元或計數固定長度向量。

### 圓形指紋（ECFP 風格）
- **ecfp** / **ecfp:2** / **ecfp:4** / **ecfp:6** - 擴展連接指紋
  - 半徑變體（2、4、6 對應直徑）
  - 預設：radius=3、2048 位元
  - 最受歡迎的相似性搜索方法
- **ecfp-count** - ECFP 的計數版本（非二元）
- **fcfp** / **fcfp-count** - 功能類別圓形指紋
  - 類似 ECFP 但使用官能基
  - 更適合基於藥效團的相似性

### 路徑指紋
- **rdkit** - 基於線性路徑的 RDKit 拓撲指紋
- **pattern** - 模式指紋（類似 MACCS 但自動化）
- **layered** - 具有多個子結構層的分層指紋

### 鍵式指紋
- **maccs** - MACCS 鍵（166 位元結構鍵）
  - 預定義子結構的固定集合
  - 適合骨架跳躍
  - 計算快速
- **avalon** - Avalon 指紋
  - 類似 MACCS 但更多特徵
  - 針對相似性搜索優化

### 原子對指紋
- **atompair** - 原子對指紋
  - 編碼原子對及其之間的距離
  - 適合 3D 相似性
- **atompair-count** - 原子對的計數版本

### 拓撲扭轉指紋
- **topological** - 拓撲扭轉指紋
  - 編碼 4 個連接原子的序列
  - 捕捉局部拓撲
- **topological-count** - 拓撲扭轉的計數版本

### 最小雜湊指紋
- **map4** - 最小雜湊原子對指紋（最多 4 個鍵）
  - 結合原子對和 ECFP 概念
  - 預設：1024 維
  - 適合大型資料集的快速高效方法
- **secfp** - SMILES 擴展連接指紋
  - 直接在 SMILES 字串上操作
  - 同時捕捉子結構和原子對資訊

### 擴展簡化圖
- **erg** - 擴展簡化圖
  - 使用藥效團點而非原子
  - 在保留關鍵特徵的同時降低圖複雜度

## 藥效團描述符

基於藥理相關官能基及其空間關係的特徵。

### CATS（化學高級模板搜索）
- **cats2D** - 2D CATS 描述符
  - 藥效團點對分佈
  - 基於最短路徑的距離
  - 預設 21 個描述符
- **cats3D** - 3D CATS 描述符
  - 基於歐幾里得距離
  - 需要構象生成
- **cats2D_pharm** / **cats3D_pharm** - 藥效團變體

### Gobbi 藥效團
- **gobbi2D** - 2D 藥效團指紋
  - 8 種藥效團特徵類型：
    - 疏水性
    - 芳香性
    - 氫鍵受體
    - 氫鍵供體
    - 正離子化
    - 負離子化
    - 集合疏水性
  - 適合虛擬篩選

### Pmapper 藥效團
- **pmapper2D** - 2D 藥效團簽名
- **pmapper3D** - 3D 藥效團簽名
  - 高維藥效團描述符
  - 用於 QSAR 和相似性搜索

## 形狀描述符

捕捉 3D 分子形狀和靜電性質的描述符。

### USR（超快形狀識別）
- **usr** - 基本 USR 描述符
  - 12 維編碼形狀分佈
  - 計算極快
- **usrcat** - 帶藥效團約束的 USR
  - 60 維（每種特徵類型 12 維）
  - 結合形狀和藥效團資訊

### 電形狀
- **electroshape** - 電形狀描述符
  - 結合分子形狀、手性和靜電
  - 用於蛋白質-配體對接預測

## 骨架描述符

基於分子骨架和核心結構的描述符。

### 骨架鍵
- **scaffoldkeys** - 骨架鍵計算器
  - 40+ 種基於骨架的屬性
  - 生物等排骨架表示
  - 捕捉核心結構特徵

## GNN 輸入的圖形特徵提取器

用於為圖神經網路構建圖表示的原子和鍵級特徵。

### 原子級特徵
- **atom-onehot** - 獨熱編碼的原子特徵
- **atom-default** - 預設原子特徵化，包括：
  - 原子序數
  - 度數、形式電荷
  - 雜化
  - 芳香性
  - 氫原子數

### 鍵級特徵
- **bond-onehot** - 獨熱編碼的鍵特徵
- **bond-default** - 預設鍵特徵化，包括：
  - 鍵類型（單鍵、雙鍵、三鍵、芳香鍵）
  - 共軛性
  - 環成員資格
  - 立體化學

## 整合預訓練模型集合

Molfeat 整合了來自各種來源的模型：

### HuggingFace 模型
透過 HuggingFace hub 存取 transformer 模型：
- ChemBERTa 變體
- ChemGPT 變體
- MolT5
- 自訂上傳的模型

### DGL-LifeSci 模型
來自 DGL-Life 的預訓練 GNN 模型：
- 具有不同預訓練任務的 GIN 變體
- AttentiveFP 模型
- MPNN 模型

### FCD（Fréchet ChemNet 距離）
- **fcd** - 用於分子生成評估的預訓練 CNN

### Graphormer 模型
- 來自 Microsoft Research 的圖 transformer
- 在量子化學資料集上預訓練

## 使用說明

### 選擇特徵提取器

**傳統 ML（隨機森林、SVM 等）：**
- 從 **ecfp** 或 **maccs** 指紋開始
- 對可解釋模型嘗試 **desc2D**
- 使用 **FeatConcat** 組合多個指紋

**深度學習：**
- 使用 **ChemBERTa** 或 **ChemGPT** 獲取 transformer 嵌入
- 使用 **gin-supervised-*** 獲取圖神經網路嵌入
- 考慮 **Graphormer** 用於量子屬性預測

**相似性搜索：**
- **ecfp** - 通用，最受歡迎
- **maccs** - 快速，適合骨架跳躍
- **map4** - 適合大規模搜索
- **usr** / **usrcat** - 3D 形狀相似性

**藥效團方法：**
- **fcfp** - 基於官能基
- **cats2D/3D** - 藥效團對分佈
- **gobbi2D** - 明確藥效團特徵

**可解釋性：**
- **desc2D** / **mordred** - 命名描述符
- **maccs** - 可解釋的子結構鍵
- **scaffoldkeys** - 骨架特徵

### 模型依賴

某些特徵提取器需要可選依賴：

- **DGL 模型**（gin-*、jtvae）：`pip install "molfeat[dgl]"`
- **Graphormer**：`pip install "molfeat[graphormer]"`
- **Transformers**（ChemBERTa、ChemGPT、MolT5）：`pip install "molfeat[transformer]"`
- **FCD**：`pip install "molfeat[fcd]"`
- **MAP4**：`pip install "molfeat[map4]"`
- **所有依賴**：`pip install "molfeat[all]"`

### 存取所有可用模型

```python
from molfeat.store.modelstore import ModelStore

store = ModelStore()
all_models = store.available_models

# 列印所有可用的特徵提取器
for model in all_models:
    print(f"{model.name}: {model.description}")

# 搜索特定類型
transformers = [m for m in all_models if "transformer" in m.tags]
gnn_models = [m for m in all_models if "gnn" in m.tags]
fingerprints = [m for m in all_models if "fingerprint" in m.tags]
```

## 效能特性

### 計算速度（相對）
**最快：**
- maccs
- ecfp
- rdkit 指紋
- usr

**中等：**
- desc2D
- cats2D
- 大多數指紋

**較慢：**
- mordred（1800+ 描述符）
- desc3D（需要構象生成）
- 一般 3D 描述符

**最慢（首次運行）：**
- 預訓練模型（ChemBERTa、ChemGPT、GIN）
- 注意：後續運行受益於快取

### 維度

**低（< 200 維）：**
- maccs（167）
- usr（12）
- usrcat（60）

**中等（200-2000 維）：**
- desc2D（約 200）
- ecfp（預設 2048，可配置）
- map4（預設 1024）

**高（> 2000 維）：**
- mordred（1800+）
- 連接的指紋
- 某些 transformer 嵌入

**可變：**
- Transformer 模型（通常 768-1024）
- GNN 模型（取決於架構）
