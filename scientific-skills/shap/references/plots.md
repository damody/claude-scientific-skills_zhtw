# SHAP 視覺化參考

本文件提供所有 SHAP 繪圖函數、其參數、使用案例以及視覺化模型解釋最佳實踐的完整資訊。

## 概述

SHAP 提供多樣化的視覺化工具，用於在個別和全局層面解釋模型預測。每種繪圖類型在理解特徵重要性、互動和預測機制方面有其特定用途。

## 繪圖類型

### 瀑布圖（Waterfall Plots）

**用途**：顯示個別預測的解釋，展示每個特徵如何將預測從基準值（期望值）推向最終預測。

**函數**：`shap.plots.waterfall(explanation, max_display=10, show=True)`

**關鍵參數**：
- `explanation`：來自 Explanation 物件的單行（非多個樣本）
- `max_display`：要顯示的特徵數量（預設：10）；影響較小的特徵會合併為單一「其他特徵」項
- `show`：是否立即顯示繪圖

**視覺元素**：
- **X 軸**：顯示 SHAP 值（對預測的貢獻）
- **起點**：模型的期望值（基準值）
- **特徵貢獻**：紅色條（正值）或藍色條（負值）顯示每個特徵如何移動預測
- **特徵值**：以灰色顯示在特徵名稱左側
- **終點**：最終模型預測

**何時使用**：
- 詳細解釋個別預測
- 理解哪些特徵驅動特定決策
- 為單一實例傳達模型行為（例如，貸款拒絕、診斷）
- 除錯意外預測

**重要說明**：
- 對於 XGBoost 分類器，預測以對數賠率單位解釋（logistic 轉換前的邊際輸出）
- SHAP 值總和等於基準值和最終預測之間的差異（可加性屬性）
- 使用散佈圖配合瀑布圖來探索多個樣本的模式

**範例**：
```python
import shap

# 計算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 繪製第一個預測的瀑布圖
shap.plots.waterfall(shap_values[0])

# 顯示更多特徵
shap.plots.waterfall(shap_values[0], max_display=20)
```

### 蜂群圖（Beeswarm Plots）

**用途**：資訊密集的摘要，展示頂部特徵如何在整個資料集上影響模型輸出，結合特徵重要性和值分佈。

**函數**：`shap.plots.beeswarm(shap_values, max_display=10, order=Explanation.abs.mean(0), color=None, show=True)`

**關鍵參數**：
- `shap_values`：包含多個樣本的 Explanation 物件
- `max_display`：要顯示的特徵數量（預設：10）
- `order`：如何對特徵進行排名
  - `Explanation.abs.mean(0)`：平均絕對 SHAP 值（預設）
  - `Explanation.abs.max(0)`：最大絕對值（突出離群值影響）
- `color`：matplotlib 色彩圖；預設為紅藍配色
- `show`：是否立即顯示繪圖

**視覺元素**：
- **Y 軸**：按重要性排名的特徵
- **X 軸**：SHAP 值（對模型輸出的影響）
- **每個點**：資料集中的單一實例
- **點位置（X）**：SHAP 值大小
- **點顏色**：原始特徵值（紅色 = 高，藍色 = 低）
- **點聚集**：顯示影響的密度/分佈

**何時使用**：
- 總結整個資料集的特徵重要性
- 理解平均和個別特徵影響
- 識別特徵值模式及其效果
- 比較不同特徵的全局模型行為
- 檢測非線性關係（例如，較高年齡 → 較低收入可能性）

**實用變體**：
```python
# 標準蜂群圖
shap.plots.beeswarm(shap_values)

# 顯示更多特徵
shap.plots.beeswarm(shap_values, max_display=20)

# 按最大絕對值排序（突出離群值）
shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))

# 繪製絕對 SHAP 值並使用固定著色
shap.plots.beeswarm(shap_values.abs, color="shap_red")

# 自訂 matplotlib 色彩圖
shap.plots.beeswarm(shap_values, color=plt.cm.viridis)
```

### 長條圖（Bar Plots）

**用途**：以平均絕對 SHAP 值顯示特徵重要性，提供全局特徵影響的清晰、簡單視覺化。

**函數**：`shap.plots.bar(shap_values, max_display=10, clustering=None, clustering_cutoff=0.5, show=True)`

**關鍵參數**：
- `shap_values`：Explanation 物件（可以是單一實例、全局或群組）
- `max_display`：要顯示的最大特徵/長條數
- `clustering`：來自 `shap.utils.hclust` 的可選階層聚類物件
- `clustering_cutoff`：顯示聚類結構的閾值（0-1，預設：0.5）

**繪圖類型**：

#### 全局長條圖
顯示所有樣本的整體特徵重要性。重要性計算為平均絕對 SHAP 值。

```python
# 全局特徵重要性
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.bar(shap_values)
```

#### 局部長條圖
顯示單一實例的 SHAP 值，特徵值以灰色顯示。

```python
# 單一預測解釋
shap.plots.bar(shap_values[0])
```

#### 群組長條圖
透過傳遞 Explanation 物件的字典來比較子群組的特徵重要性。

```python
# 比較群組
cohorts = {
    "Group A": shap_values[mask_A],
    "Group B": shap_values[mask_B]
}
shap.plots.bar(cohorts)
```

**特徵聚類**：
使用基於模型的聚類識別冗餘特徵（比基於相關性的方法更準確）。

```python
# 添加特徵聚類
clustering = shap.utils.hclust(X_train, y_train)
shap.plots.bar(shap_values, clustering=clustering)

# 調整聚類顯示閾值
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.3)
```

**何時使用**：
- 快速概覽全局特徵重要性
- 比較群組或模型間的特徵重要性
- 識別冗餘或相關特徵
- 簡報用的清晰、簡單視覺化

### 力場圖（Force Plots）

**用途**：加性力場視覺化，顯示特徵如何將預測從基準值推高（紅色）或推低（藍色）。

**函數**：`shap.plots.force(base_value, shap_values, features, feature_names=None, out_names=None, link="identity", matplotlib=False, show=True)`

**關鍵參數**：
- `base_value`：期望值（基準預測）
- `shap_values`：樣本的 SHAP 值
- `features`：樣本的特徵值
- `feature_names`：可選的特徵名稱
- `link`：轉換函數（"identity" 或 "logit"）
- `matplotlib`：使用 matplotlib 後端（預設：互動 JavaScript）

**視覺元素**：
- **基準值**：起始預測（期望值）
- **紅色箭頭**：將預測推高的特徵
- **藍色箭頭**：將預測推低的特徵
- **最終值**：結果預測

**互動功能**（JavaScript 模式）：
- 懸停顯示詳細特徵資訊
- 多個樣本建立堆疊視覺化
- 可旋轉以獲得不同視角

**何時使用**：
- 互動探索預測
- 同時視覺化多個預測
- 需要互動元素的簡報
- 一目了然地理解預測組成

**範例**：
```python
# 單一預測力場圖
shap.plots.force(
    shap_values.base_values[0],
    shap_values.values[0],
    X_test.iloc[0],
    matplotlib=True
)

# 多個預測（互動）
shap.plots.force(
    shap_values.base_values,
    shap_values.values,
    X_test
)
```

### 散佈圖（Scatter Plots / 依賴圖）

**用途**：顯示特徵值和其 SHAP 值之間的關係，揭示特徵值如何影響預測。

**函數**：`shap.plots.scatter(shap_values, color=None, hist=True, alpha=1, show=True)`

**關鍵參數**：
- `shap_values`：Explanation 物件，可以用下標指定特徵（例如，`shap_values[:, "Age"]`）
- `color`：用於著色點的特徵（字串名稱或 Explanation 物件）
- `hist`：在 y 軸顯示特徵值的直方圖
- `alpha`：點透明度（對密集繪圖有用）

**視覺元素**：
- **X 軸**：特徵值
- **Y 軸**：SHAP 值（對預測的影響）
- **點顏色**：另一個特徵的值（用於互動檢測）
- **直方圖**：特徵值的分佈

**何時使用**：
- 理解特徵-預測關係
- 檢測非線性效應
- 識別特徵互動
- 驗證或發現模型行為中的模式
- 探索瀑布圖中的反直覺預測

**互動檢測**：
按另一個特徵著色點以揭示互動。

```python
# 基本依賴圖
shap.plots.scatter(shap_values[:, "Age"])

# 按另一個特徵著色以顯示互動
shap.plots.scatter(shap_values[:, "Age"], color=shap_values[:, "Education"])

# 一張圖中顯示多個特徵
shap.plots.scatter(shap_values[:, ["Age", "Education", "Hours-per-week"]])

# 增加密集資料的透明度
shap.plots.scatter(shap_values[:, "Age"], alpha=0.5)
```

### 熱圖（Heatmap Plots）

**用途**：同時視覺化多個樣本的 SHAP 值，顯示跨實例的特徵影響。

**函數**：`shap.plots.heatmap(shap_values, instance_order=None, feature_values=None, max_display=10, show=True)`

**關鍵參數**：
- `shap_values`：Explanation 物件
- `instance_order`：如何排序實例（可以是自訂排序的 Explanation 物件）
- `feature_values`：懸停時顯示特徵值
- `max_display`：要顯示的最大特徵數

**視覺元素**：
- **行**：個別實例/樣本
- **列**：特徵
- **儲存格顏色**：SHAP 值（紅色 = 正值，藍色 = 負值）
- **強度**：影響大小

**何時使用**：
- 比較多個實例的解釋
- 識別特徵影響的模式
- 理解哪些特徵在預測間變化最大
- 檢測具有相似解釋模式的子群組或聚類

**範例**：
```python
# 基本熱圖
shap.plots.heatmap(shap_values)

# 按模型輸出排序實例
shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1))

# 顯示特定子集
shap.plots.heatmap(shap_values[:100])
```

### 小提琴圖（Violin Plots）

**用途**：類似蜂群圖，但使用小提琴（核密度）視覺化而非個別點。

**函數**：`shap.plots.violin(shap_values, features=None, feature_names=None, max_display=10, show=True)`

**何時使用**：
- 當資料集非常大時作為蜂群圖的替代
- 強調分佈密度而非個別點
- 簡報用的更清晰視覺化

**範例**：
```python
shap.plots.violin(shap_values)
```

### 決策圖（Decision Plots）

**用途**：透過累積 SHAP 值顯示預測路徑，特別適用於多類別分類。

**函數**：`shap.plots.decision(base_value, shap_values, features, feature_names=None, feature_order="importance", highlight=None, link="identity", show=True)`

**關鍵參數**：
- `base_value`：期望值
- `shap_values`：樣本的 SHAP 值
- `features`：特徵值
- `feature_order`：如何排序特徵（"importance" 或列表）
- `highlight`：要突出顯示的樣本索引
- `link`：轉換函數

**何時使用**：
- 多類別分類解釋
- 理解累積特徵效應
- 比較跨樣本的預測路徑
- 識別預測分歧的位置

**範例**：
```python
# 多個預測的決策圖
shap.plots.decision(
    shap_values.base_values,
    shap_values.values,
    X_test,
    feature_names=X_test.columns.tolist()
)

# 突出顯示特定實例
shap.plots.decision(
    shap_values.base_values,
    shap_values.values,
    X_test,
    highlight=[0, 5, 10]
)
```

## 繪圖選擇指南

**用於個別預測**：
- **瀑布圖**：最適合詳細、順序解釋
- **力場圖**：適合互動探索
- **長條圖（局部）**：簡單、清晰的單一預測重要性

**用於全局理解**：
- **蜂群圖**：具有值分佈的資訊密集摘要
- **長條圖（全局）**：清晰、簡單的重要性排名
- **小提琴圖**：蜂群圖的分佈導向替代

**用於特徵關係**：
- **散佈圖**：理解特徵-預測關係和互動
- **熱圖**：比較多個實例的模式

**用於多個樣本**：
- **熱圖**：SHAP 值的網格視圖
- **力場圖（堆疊）**：互動多樣本視覺化
- **決策圖**：多類別問題的預測路徑

**用於群組比較**：
- **長條圖（群組）**：清晰的特徵重要性比較
- **多個蜂群圖**：並排分佈比較

## 視覺化最佳實踐

**1. 先全局，後局部**：
- 從蜂群圖或長條圖開始理解全局模式
- 深入瀑布圖或散佈圖研究特定實例或特徵

**2. 使用多種繪圖類型**：
- 不同繪圖揭示不同見解
- 結合瀑布圖（個別）+ 散佈圖（關係）+ 蜂群圖（全局）

**3. 調整 max_display**：
- 預設（10）適合簡報
- 增加（20-30）用於詳細分析
- 考慮對冗餘特徵使用聚類

**4. 有意義地使用顏色**：
- 對 SHAP 值使用預設紅藍色（紅色 = 正值，藍色 = 負值）
- 用互動特徵對散佈圖著色
- 特定領域使用自訂色彩圖

**5. 考慮觀眾**：
- 技術觀眾：蜂群圖、散佈圖、熱圖
- 非技術觀眾：瀑布圖、長條圖、力場圖
- 互動簡報：使用 JavaScript 的力場圖

**6. 儲存高品質圖形**：
```python
import matplotlib.pyplot as plt

# 建立繪圖
shap.plots.beeswarm(shap_values, show=False)

# 以高 DPI 儲存
plt.savefig('shap_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

**7. 處理大型資料集**：
- 對視覺化取樣子集（例如，`shap_values[:1000]`）
- 對非常大的資料集使用小提琴圖而非蜂群圖
- 對有許多點的散佈圖調整 alpha

## 常見模式和工作流程

**模式 1：完整模型解釋**
```python
# 1. 全局重要性
shap.plots.beeswarm(shap_values)

# 2. 頂部特徵關係
for feature in top_features:
    shap.plots.scatter(shap_values[:, feature])

# 3. 範例預測
for i in interesting_indices:
    shap.plots.waterfall(shap_values[i])
```

**模式 2：模型比較**
```python
# 計算多個模型的 SHAP
shap_model1 = explainer1(X_test)
shap_model2 = explainer2(X_test)

# 比較特徵重要性
shap.plots.bar({
    "Model 1": shap_model1,
    "Model 2": shap_model2
})
```

**模式 3：子群組分析**
```python
# 定義群組
male_mask = X_test['Sex'] == 'Male'
female_mask = X_test['Sex'] == 'Female'

# 比較群組
shap.plots.bar({
    "Male": shap_values[male_mask],
    "Female": shap_values[female_mask]
})

# 分開的蜂群圖
shap.plots.beeswarm(shap_values[male_mask])
shap.plots.beeswarm(shap_values[female_mask])
```

**模式 4：除錯預測**
```python
# 識別離群值或錯誤
errors = (model.predict(X_test) != y_test)
error_indices = np.where(errors)[0]

# 解釋錯誤
for idx in error_indices[:5]:
    print(f"Sample {idx}:")
    shap.plots.waterfall(shap_values[idx])

    # 探索關鍵特徵
    shap.plots.scatter(shap_values[:, "Key_Feature"])
```

## 與 Notebooks 和報告的整合

**Jupyter Notebooks**：
- 互動力場圖無縫運作
- 使用 `show=True`（預設）進行內嵌顯示
- 與 markdown 解釋結合

**靜態報告**：
- 力場圖使用 matplotlib 後端
- 程式化儲存圖形
- 清晰度優先使用瀑布圖和長條圖

**網頁應用程式**：
- 將力場圖匯出為 HTML
- 使用 shap.save_html() 進行互動視覺化
- 考慮按需產生繪圖

## 視覺化疑難排解

**問題**：繪圖不顯示
- **解決方案**：確保正確設定 matplotlib 後端；如需要使用 `plt.show()`

**問題**：太多特徵使繪圖混亂
- **解決方案**：減少 `max_display` 參數或使用特徵聚類

**問題**：顏色反轉或令人困惑
- **解決方案**：檢查模型輸出類型（機率 vs. 對數賠率）並使用適當的連結函數

**問題**：大型資料集繪圖緩慢
- **解決方案**：對資料子集取樣；使用 `shap_values[:1000]` 進行視覺化

**問題**：缺少特徵名稱
- **解決方案**：確保特徵名稱在 Explanation 物件中或明確傳遞給繪圖函數
