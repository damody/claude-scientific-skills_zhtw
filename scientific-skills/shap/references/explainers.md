# SHAP 解釋器參考

本文件提供所有 SHAP 解釋器類別、其參數、方法以及何時使用每種類型的完整資訊。

## 概述

SHAP 為不同模型類型提供專門的解釋器，每個都針對特定架構進行優化。通用的 `shap.Explainer` 類別會根據模型類型自動選擇適當的演算法。

## 核心解釋器類別

### shap.Explainer（自動選擇器）

**用途**：透過選擇最適當的解釋器演算法，自動使用 Shapley 值來解釋任何機器學習模型或 Python 函數。

**建構函數參數**：
- `model`：要解釋的模型（函數或模型物件）
- `masker`：用於特徵操作的背景資料或遮罩器物件
- `algorithm`：可選覆寫以強制使用特定解釋器類型
- `output_names`：模型輸出的名稱
- `feature_names`：輸入特徵的名稱

**何時使用**：當不確定使用哪個解釋器時的預設選擇；根據模型類型自動選擇最佳演算法。

### TreeExplainer

**用途**：使用 Tree SHAP 演算法為基於樹的集成模型進行快速且精確的 SHAP 值計算。

**建構函數參數**：
- `model`：基於樹的模型（XGBoost、LightGBM、CatBoost、PySpark 或 scikit-learn 樹）
- `data`：用於特徵整合的背景資料集（使用 tree_path_dependent 時為可選）
- `feature_perturbation`：如何處理依賴特徵
  - `"interventional"`：需要背景資料；遵循因果推論規則
  - `"tree_path_dependent"`：不需要背景資料；使用每個葉節點的訓練樣本
  - `"auto"`：如果提供資料則預設為 interventional，否則為 tree_path_dependent
- `model_output`：要解釋的模型輸出
  - `"raw"`：標準模型輸出（預設）
  - `"probability"`：機率轉換輸出
  - `"log_loss"`：損失函數的自然對數
  - 自訂方法名稱如 `"predict_proba"`
- `feature_names`：可選的特徵命名

**支援的模型**：
- XGBoost（xgboost.XGBClassifier、xgboost.XGBRegressor、xgboost.Booster）
- LightGBM（lightgbm.LGBMClassifier、lightgbm.LGBMRegressor、lightgbm.Booster）
- CatBoost（catboost.CatBoostClassifier、catboost.CatBoostRegressor）
- PySpark MLlib 樹模型
- scikit-learn（DecisionTreeClassifier、DecisionTreeRegressor、RandomForestClassifier、RandomForestRegressor、ExtraTreesClassifier、ExtraTreesRegressor、GradientBoostingClassifier、GradientBoostingRegressor）

**關鍵方法**：
- `shap_values(X)`：計算樣本的 SHAP 值；返回每行代表特徵歸因的陣列
- `shap_interaction_values(X)`：估計特徵對之間的互動效應；提供具有主效應和成對互動的矩陣
- `explain_row(row)`：解釋單行並提供詳細的歸因資訊

**何時使用**：
- 所有基於樹模型的首選
- 當需要精確 SHAP 值（非近似值）時
- 當大型資料集的計算速度很重要時
- 對於隨機森林、梯度提升或 XGBoost 等模型

**範例**：
```python
import shap
import xgboost

# 訓練模型
model = xgboost.XGBClassifier().fit(X_train, y_train)

# 建立解釋器
explainer = shap.TreeExplainer(model)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test)

# 計算互動值
shap_interaction = explainer.shap_interaction_values(X_test)
```

### DeepExplainer

**用途**：使用增強版 DeepLIFT 演算法為深度學習模型近似 SHAP 值。

**建構函數參數**：
- `model`：依框架而定的規格
  - **TensorFlow**：(input_tensor, output_tensor) 的元組，其中輸出為單維
  - **PyTorch**：`nn.Module` 物件或 `(model, layer)` 的元組用於特定層的解釋
- `data`：用於特徵整合的背景資料集
  - **TensorFlow**：numpy 陣列或 pandas DataFrames
  - **PyTorch**：torch tensors
  - **建議大小**：100-1000 個樣本（非完整訓練集）以平衡準確性和計算成本
- `session`（僅限 TensorFlow）：可選的會話物件；如果為 None 則自動檢測
- `learning_phase_flags`：用於處理推論期間 batch norm/dropout 的自訂學習階段張量

**支援的框架**：
- **TensorFlow**：完整支援包括 Keras 模型
- **PyTorch**：與 nn.Module 架構完整整合

**關鍵方法**：
- `shap_values(X)`：返回應用於資料 X 的模型的近似 SHAP 值
- `explain_row(row)`：解釋單行並提供歸因值和期望輸出
- `save(file)` / `load(file)`：解釋器物件的序列化支援
- `supports_model_with_masker(model, masker)`：模型類型的相容性檢查器

**何時使用**：
- 用於 TensorFlow 或 PyTorch 中的深度神經網路
- 處理卷積神經網路（CNNs）時
- 用於循環神經網路（RNNs）和 transformers
- 當深度學習架構需要模型特定解釋時

**關鍵設計特徵**：
期望估計的變異數大約以 1/√N 的比例縮放，其中 N 是背景樣本數量，實現準確性-效率的權衡。

**範例**：
```python
import shap
import tensorflow as tf

# 假設 model 是一個 Keras 模型
model = tf.keras.models.load_model('my_model.h5')

# 選擇背景樣本（訓練資料的子集）
background = X_train[:100]

# 建立解釋器
explainer = shap.DeepExplainer(model, background)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test[:10])
```

### KernelExplainer

**用途**：使用 Kernel SHAP 方法與加權線性迴歸進行模型無關的 SHAP 值計算。

**建構函數參數**：
- `model`：接受樣本矩陣並返回模型輸出的函數或模型物件
- `data`：用於模擬缺失特徵的背景資料集（numpy 陣列、pandas DataFrame 或稀疏矩陣）
- `feature_names`：可選的特徵名稱列表；如果可用則自動從 DataFrame 欄位名稱衍生
- `link`：特徵重要性和模型輸出之間的連接函數
  - `"identity"`：直接關係（預設）
  - `"logit"`：用於機率輸出

**關鍵方法**：
- `shap_values(X, **kwargs)`：計算樣本預測的 SHAP 值
  - `nsamples`：每個預測的評估計數（"auto" 或整數）；較高的值可減少變異
  - `l1_reg`：特徵選擇正則化（"num_features(int)"、"aic"、"bic" 或浮點數）
  - 返回每行總和等於模型輸出與期望值之差的陣列
- `explain_row(row)`：解釋個別預測並提供歸因值和期望值
- `save(file)` / `load(file)`：持久化和恢復解釋器物件

**何時使用**：
- 用於沒有專門解釋器可用的黑盒模型
- 處理自訂預測函數時
- 用於任何模型類型（神經網路、SVMs、集成方法等）
- 當需要模型無關的解釋時
- **注意**：比專門的解釋器慢；僅在沒有專門選項存在時使用

**範例**：
```python
import shap
from sklearn.svm import SVC

# 訓練模型
model = SVC(probability=True).fit(X_train, y_train)

# 建立預測函數
predict_fn = lambda x: model.predict_proba(x)[:, 1]

# 選擇背景樣本
background = shap.sample(X_train, 100)

# 建立解釋器
explainer = shap.KernelExplainer(predict_fn, background)

# 計算 SHAP 值（可能較慢）
shap_values = explainer.shap_values(X_test[:10])
```

### LinearExplainer

**用途**：用於線性模型的專門解釋器，考慮特徵相關性。

**建構函數參數**：
- `model`：線性模型或（係數、截距）的元組
- `masker`：用於特徵相關性的背景資料
- `feature_perturbation`：如何處理特徵相關性
  - `"interventional"`：假設特徵獨立
  - `"correlation_dependent"`：考慮特徵相關性

**支援的模型**：
- scikit-learn 線性模型（LinearRegression、LogisticRegression、Ridge、Lasso、ElasticNet）
- 具有係數和截距的自訂線性模型

**何時使用**：
- 用於線性迴歸和邏輯迴歸模型
- 當特徵相關性對解釋準確性很重要時
- 當需要極快速度的解釋時
- 用於 GLMs 和其他線性模型類型

**範例**：
```python
import shap
from sklearn.linear_model import LogisticRegression

# 訓練模型
model = LogisticRegression().fit(X_train, y_train)

# 建立解釋器
explainer = shap.LinearExplainer(model, X_train)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test)
```

### GradientExplainer

**用途**：使用期望梯度來近似神經網路的 SHAP 值。

**建構函數參數**：
- `model`：深度學習模型（TensorFlow 或 PyTorch）
- `data`：用於整合的背景樣本
- `batch_size`：梯度計算的批次大小
- `local_smoothing`：要添加用於平滑的雜訊量（預設 0）

**何時使用**：
- 作為神經網路 DeepExplainer 的替代方案
- 當偏好基於梯度的解釋時
- 用於梯度資訊可用的可微分模型

**範例**：
```python
import shap
import torch

# 假設 model 是一個 PyTorch 模型
model = torch.load('model.pt')

# 選擇背景樣本
background = X_train[:100]

# 建立解釋器
explainer = shap.GradientExplainer(model, background)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test[:10])
```

### PermutationExplainer

**用途**：透過迭代輸入的排列來近似 Shapley 值。

**建構函數參數**：
- `model`：預測函數
- `masker`：背景資料或遮罩器物件
- `max_evals`：每個樣本的最大模型評估次數

**何時使用**：
- 當需要精確 Shapley 值但沒有專門的解釋器可用時
- 對於排列可處理的小特徵集
- 作為 KernelExplainer 更準確但更慢的替代方案

**範例**：
```python
import shap

# 建立解釋器
explainer = shap.PermutationExplainer(model.predict, X_train)

# 計算 SHAP 值
shap_values = explainer.shap_values(X_test[:10])
```

## 解釋器選擇指南

**選擇解釋器的決策樹**：

1. **您的模型是基於樹的嗎？**（XGBoost、LightGBM、CatBoost、隨機森林等）
   - 是 → 使用 `TreeExplainer`（快速且精確）
   - 否 → 繼續步驟 2

2. **您的模型是深度神經網路嗎？**（TensorFlow、PyTorch、Keras）
   - 是 → 使用 `DeepExplainer` 或 `GradientExplainer`
   - 否 → 繼續步驟 3

3. **您的模型是線性的嗎？**（線性/邏輯迴歸、GLMs）
   - 是 → 使用 `LinearExplainer`（極快）
   - 否 → 繼續步驟 4

4. **您需要模型無關的解釋嗎？**
   - 是 → 使用 `KernelExplainer`（較慢但適用於任何模型）
   - 如果計算預算允許且需要高準確性 → 使用 `PermutationExplainer`

5. **不確定或想要自動選擇？**
   - 使用 `shap.Explainer`（自動選擇最佳演算法）

## 解釋器間的通用參數

**背景資料 / 遮罩器**：
- 用途：代表「典型」輸入以建立基準期望
- 大小建議：50-1000 個樣本（複雜模型需要更多）
- 選擇：從訓練資料隨機取樣或 kmeans 選擇的代表

**特徵名稱**：
- 自動從 pandas DataFrames 提取
- 可以為 numpy 陣列手動指定
- 對繪圖可解釋性很重要

**模型輸出規格**：
- 原始模型輸出 vs. 轉換輸出（機率、對數賠率）
- 對正確解釋 SHAP 值至關重要
- 範例：對於 XGBoost 分類器，SHAP 解釋 logistic 轉換前的邊際輸出（對數賠率）

## 效能考量

**速度排名**（從最快到最慢）：
1. `LinearExplainer` - 幾乎瞬間完成
2. `TreeExplainer` - 非常快，擴展性好
3. `DeepExplainer` - 對神經網路快速
4. `GradientExplainer` - 對神經網路快速
5. `KernelExplainer` - 慢，僅在必要時使用
6. `PermutationExplainer` - 非常慢但對小特徵集最準確

**記憶體考量**：
- `TreeExplainer`：低記憶體開銷
- `DeepExplainer`：記憶體與背景樣本大小成正比
- `KernelExplainer`：大型背景資料集可能佔用大量記憶體
- 對於大型資料集：使用批次處理或樣本子集

## 解釋器輸出：解釋物件

所有解釋器返回包含以下內容的 `shap.Explanation` 物件：
- `values`：SHAP 值（numpy 陣列）
- `base_values`：期望模型輸出（基準值）
- `data`：原始特徵值
- `feature_names`：特徵名稱

解釋物件支援：
- 切片：`explanation[0]` 取得第一個樣本
- 陣列操作：與 numpy 操作相容
- 直接繪圖：可以傳遞給繪圖函數
