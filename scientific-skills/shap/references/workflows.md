# SHAP 工作流程和最佳實踐

本文件提供在各種模型解釋場景中使用 SHAP 的完整工作流程、最佳實踐和常見用例。

## 基本工作流程結構

每個 SHAP 分析都遵循一般工作流程：

1. **訓練模型**：建構和訓練機器學習模型
2. **選擇解釋器**：根據模型類型選擇適當的解釋器
3. **計算 SHAP 值**：為測試樣本產生解釋
4. **視覺化結果**：使用繪圖理解特徵影響
5. **解釋和行動**：得出結論並做出決策

## 工作流程 1：基本模型解釋

**用例**：理解訓練模型的特徵重要性和預測行為

```python
import shap
import pandas as pd
from sklearn.model_selection import train_test_split

# 步驟 1：載入和分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 步驟 2：訓練模型（以 XGBoost 為例）
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# 步驟 3：建立解釋器
explainer = shap.TreeExplainer(model)

# 步驟 4：計算 SHAP 值
shap_values = explainer(X_test)

# 步驟 5：視覺化全局重要性
shap.plots.beeswarm(shap_values, max_display=15)

# 步驟 6：詳細檢查頂部特徵
shap.plots.scatter(shap_values[:, "Feature1"])
shap.plots.scatter(shap_values[:, "Feature2"], color=shap_values[:, "Feature1"])

# 步驟 7：解釋個別預測
shap.plots.waterfall(shap_values[0])
```

**關鍵決策**：
- 根據模型架構選擇解釋器類型
- 背景資料集大小（對於 DeepExplainer、KernelExplainer）
- 要解釋的樣本數量（全部測試集 vs. 子集）

## 工作流程 2：模型除錯和驗證

**用例**：識別和修復模型問題、驗證預期行為

```python
# 步驟 1：計算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 步驟 2：識別預測錯誤
predictions = model.predict(X_test)
errors = predictions != y_test
error_indices = np.where(errors)[0]

# 步驟 3：分析錯誤
print(f"Total errors: {len(error_indices)}")
print(f"Error rate: {len(error_indices) / len(y_test):.2%}")

# 步驟 4：解釋誤分類樣本
for idx in error_indices[:10]:  # 前 10 個錯誤
    print(f"\n=== Error {idx} ===")
    print(f"Prediction: {predictions[idx]}, Actual: {y_test.iloc[idx]}")
    shap.plots.waterfall(shap_values[idx])

# 步驟 5：檢查模型是否學到正確模式
# 尋找意外的特徵重要性
shap.plots.beeswarm(shap_values)

# 步驟 6：調查特定特徵關係
# 驗證非線性關係是否合理
for feature in model.feature_importances_.argsort()[-5:]:  # 前 5 個特徵
    feature_name = X_test.columns[feature]
    shap.plots.scatter(shap_values[:, feature_name])

# 步驟 7：驗證特徵互動
# 檢查互動是否與領域知識一致
shap.plots.scatter(shap_values[:, "Feature1"], color=shap_values[:, "Feature2"])
```

**要檢查的常見問題**：
- 資料洩漏（具有可疑高重要性的特徵）
- 虛假相關性（意外的特徵關係）
- 目標洩漏（不應具有預測性的特徵）
- 偏差（對某些群組的不成比例影響）

## 工作流程 3：特徵工程指導

**用例**：使用 SHAP 見解改進特徵工程

```python
# 步驟 1：具有基準特徵的初始模型
model_v1 = train_model(X_train_v1, y_train)
explainer_v1 = shap.TreeExplainer(model_v1)
shap_values_v1 = explainer_v1(X_test_v1)

# 步驟 2：識別特徵工程機會
shap.plots.beeswarm(shap_values_v1)

# 檢查：
# - 非線性關係（轉換候選）
shap.plots.scatter(shap_values_v1[:, "Age"])  # 也許 age^2 或年齡分箱？

# - 特徵互動（互動項候選）
shap.plots.scatter(shap_values_v1[:, "Income"], color=shap_values_v1[:, "Education"])
# 也許建立 Income * Education 互動？

# 步驟 3：根據見解工程化新特徵
X_train_v2 = X_train_v1.copy()
X_train_v2['Age_squared'] = X_train_v2['Age'] ** 2
X_train_v2['Income_Education'] = X_train_v2['Income'] * X_train_v2['Education']

# 步驟 4：使用工程化特徵重新訓練
model_v2 = train_model(X_train_v2, y_train)
explainer_v2 = shap.TreeExplainer(model_v2)
shap_values_v2 = explainer_v2(X_test_v2)

# 步驟 5：比較特徵重要性
shap.plots.bar({
    "Baseline": shap_values_v1,
    "With Engineered Features": shap_values_v2
})

# 步驟 6：驗證改進
print(f"V1 Score: {model_v1.score(X_test_v1, y_test):.4f}")
print(f"V2 Score: {model_v2.score(X_test_v2, y_test):.4f}")
```

**來自 SHAP 的特徵工程見解**：
- 強非線性模式 → 嘗試轉換（log、sqrt、多項式）
- 散佈圖中的顏色編碼互動 → 建立互動項
- 聚類中的冗餘特徵 → 移除或合併
- 意外的重要性 → 調查資料品質問題

## 工作流程 4：模型比較和選擇

**用例**：比較多個模型以選擇最佳可解釋模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 步驟 1：訓練多個模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train, y_train),
    'Random Forest': RandomForestClassifier(n_estimators=100).fit(X_train, y_train),
    'XGBoost': xgb.XGBClassifier(n_estimators=100).fit(X_train, y_train)
}

# 步驟 2：為每個模型計算 SHAP 值
shap_values_dict = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values_dict[name] = explainer(X_test)

# 步驟 3：比較全局特徵重要性
shap.plots.bar(shap_values_dict)

# 步驟 4：比較模型分數
for name, model in models.items():
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.4f}")

# 步驟 5：檢查特徵重要性的一致性
for feature in X_test.columns[:5]:  # 前 5 個特徵
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, shap_vals) in enumerate(shap_values_dict.items()):
        plt.sca(axes[idx])
        shap.plots.scatter(shap_vals[:, feature], show=False)
        plt.title(f"{name} - {feature}")
    plt.tight_layout()
    plt.show()

# 步驟 6：分析跨模型的特定預測
sample_idx = 0
for name, shap_vals in shap_values_dict.items():
    print(f"\n=== {name} ===")
    shap.plots.waterfall(shap_vals[sample_idx])

# 步驟 7：基於以下決策：
# - 準確性/效能
# - 可解釋性（一致的特徵重要性）
# - 部署限制
# - 利益相關者要求
```

**模型選擇標準**：
- **準確性 vs. 可解釋性**：有時使用 SHAP 的更簡單模型更可取
- **特徵一致性**：在特徵重要性上一致的模型更可信
- **解釋品質**：清晰、可行動的解釋
- **計算成本**：TreeExplainer 比 KernelExplainer 快

## 工作流程 5：公平性和偏差分析

**用例**：檢測和分析跨人口群組的模型偏差

```python
# 步驟 1：識別受保護屬性
protected_attr = 'Gender'  # 或 'Race'、'Age_Group' 等

# 步驟 2：計算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 步驟 3：比較群組間的特徵重要性
groups = X_test[protected_attr].unique()
cohorts = {
    f"{protected_attr}={group}": shap_values[X_test[protected_attr] == group]
    for group in groups
}
shap.plots.bar(cohorts)

# 步驟 4：檢查受保護屬性是否具有高 SHAP 重要性
# （對於公平模型應該很低/零）
protected_importance = np.abs(shap_values[:, protected_attr].values).mean()
print(f"{protected_attr} mean |SHAP|: {protected_importance:.4f}")

# 步驟 5：分析每個群組的預測
for group in groups:
    mask = X_test[protected_attr] == group
    group_shap = shap_values[mask]

    print(f"\n=== {protected_attr} = {group} ===")
    print(f"Sample size: {mask.sum()}")
    print(f"Positive prediction rate: {(model.predict(X_test[mask]) == 1).mean():.2%}")

    # 視覺化
    shap.plots.beeswarm(group_shap, max_display=10)

# 步驟 6：檢查代理特徵
# 與受保護屬性相關但不應具有高重要性的特徵
# 範例：'Zip_Code' 可能是種族的代理
proxy_features = ['Zip_Code', 'Last_Name_Prefix']  # 領域特定
for feature in proxy_features:
    if feature in X_test.columns:
        importance = np.abs(shap_values[:, feature].values).mean()
        print(f"Potential proxy '{feature}' importance: {importance:.4f}")

# 步驟 7：如果發現偏差的緩解策略
# - 移除受保護屬性和代理
# - 在訓練期間添加公平性約束
# - 後處理預測以均衡結果
# - 使用不同的模型架構
```

**要檢查的公平性指標**：
- **人口統計均等**：群組間相似的正面預測率
- **機會均等**：群組間相似的真陽性率
- **特徵重要性均等**：群組間相似的特徵排名
- **受保護屬性重要性**：應該最小

## 工作流程 6：深度學習模型解釋

**用例**：使用 DeepExplainer 解釋神經網路預測

```python
import tensorflow as tf
import shap

# 步驟 1：載入或建構神經網路
model = tf.keras.models.load_model('my_model.h5')

# 步驟 2：選擇背景資料集
# 使用訓練資料的子集（100-1000 個樣本）
background = X_train[:100]

# 步驟 3：建立 DeepExplainer
explainer = shap.DeepExplainer(model, background)

# 步驟 4：計算 SHAP 值（可能需要時間）
# 解釋測試資料的子集
test_subset = X_test[:50]
shap_values = explainer.shap_values(test_subset)

# 步驟 5：處理多輸出模型
# 對於二元分類，shap_values 是列表 [class_0_values, class_1_values]
# 對於迴歸，它是單一陣列
if isinstance(shap_values, list):
    # 專注於正類
    shap_values_positive = shap_values[1]
    shap_exp = shap.Explanation(
        values=shap_values_positive,
        base_values=explainer.expected_value[1],
        data=test_subset
    )
else:
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=test_subset
    )

# 步驟 6：視覺化
shap.plots.beeswarm(shap_exp)
shap.plots.waterfall(shap_exp[0])

# 步驟 7：對於圖像/文字資料，使用專門的繪圖
# 圖像：shap.image_plot
# 文字：shap.plots.text（用於 transformers）
```

**深度學習考量**：
- 背景資料集大小影響準確性和速度
- 多輸出處理（分類 vs. 迴歸）
- 圖像/文字資料的專門繪圖
- 計算成本（考慮 GPU 加速）

## 工作流程 7：生產部署

**用例**：將 SHAP 解釋整合到生產系統中

```python
import joblib
import shap

# 步驟 1：訓練並儲存模型
model = train_model(X_train, y_train)
joblib.dump(model, 'model.pkl')

# 步驟 2：建立並儲存解釋器
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, 'explainer.pkl')

# 步驟 3：建立解釋服務
class ExplanationService:
    def __init__(self, model_path, explainer_path):
        self.model = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path)

    def predict_with_explanation(self, X):
        """
        返回預測和解釋
        """
        # 預測
        prediction = self.model.predict(X)

        # SHAP 值
        shap_values = self.explainer(X)

        # 格式化解釋
        explanations = []
        for i in range(len(X)):
            exp = {
                'prediction': prediction[i],
                'base_value': shap_values.base_values[i],
                'shap_values': dict(zip(X.columns, shap_values.values[i])),
                'feature_values': X.iloc[i].to_dict()
            }
            explanations.append(exp)

        return explanations

    def get_top_features(self, X, n=5):
        """
        返回每個預測的頂部 N 個特徵
        """
        shap_values = self.explainer(X)

        top_features = []
        for i in range(len(X)):
            # 獲取絕對 SHAP 值
            abs_shap = np.abs(shap_values.values[i])

            # 排序並獲取頂部 N
            top_indices = abs_shap.argsort()[-n:][::-1]
            top_feature_names = X.columns[top_indices].tolist()
            top_shap_values = shap_values.values[i][top_indices].tolist()

            top_features.append({
                'features': top_feature_names,
                'shap_values': top_shap_values
            })

        return top_features

# 步驟 4：在 API 中使用
service = ExplanationService('model.pkl', 'explainer.pkl')

# 範例 API 端點
def predict_endpoint(input_data):
    X = pd.DataFrame([input_data])
    explanations = service.predict_with_explanation(X)
    return {
        'prediction': explanations[0]['prediction'],
        'explanation': explanations[0]
    }

# 步驟 5：為批次預測產生靜態解釋
def batch_explain_and_save(X_batch, output_dir):
    shap_values = explainer(X_batch)

    # 儲存全局繪圖
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(f'{output_dir}/global_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 儲存個別解釋
    for i in range(min(100, len(X_batch))):  # 前 100 個
        shap.plots.waterfall(shap_values[i], show=False)
        plt.savefig(f'{output_dir}/explanation_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
```

**生產最佳實踐**：
- 快取解釋器以避免重新計算
- 盡可能批次處理解釋
- 限制解釋複雜度（頂部 N 個特徵）
- 監控解釋延遲
- 將解釋器與模型一起版本控制
- 考慮為常見輸入預先計算解釋

## 工作流程 8：時間序列模型解釋

**用例**：解釋時間序列預測模型

```python
# 步驟 1：準備具有時間基礎特徵的資料
# 範例：預測下一天的銷售
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Lag_1'] = df['Sales'].shift(1)
df['Lag_7'] = df['Sales'].shift(7)
df['Rolling_Mean_7'] = df['Sales'].rolling(7).mean()

# 步驟 2：訓練模型
features = ['DayOfWeek', 'Month', 'Lag_1', 'Lag_7', 'Rolling_Mean_7']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Sales'])
model = xgb.XGBRegressor().fit(X_train, y_train)

# 步驟 3：計算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 步驟 4：分析時間模式
# 哪些特徵在不同時間驅動預測？
shap.plots.beeswarm(shap_values)

# 步驟 5：檢查滯後特徵重要性
# 滯後特徵應該對時間序列有高重要性
lag_features = ['Lag_1', 'Lag_7', 'Rolling_Mean_7']
for feature in lag_features:
    shap.plots.scatter(shap_values[:, feature])

# 步驟 6：解釋特定預測
# 例如，為什麼週一的預測如此不同？
monday_mask = X_test['DayOfWeek'] == 0
shap.plots.waterfall(shap_values[monday_mask][0])

# 步驟 7：驗證季節性理解
shap.plots.scatter(shap_values[:, 'Month'])
```

**時間序列考量**：
- 滯後特徵及其重要性
- 滾動統計的解釋
- SHAP 值中的季節性模式
- 避免特徵工程中的資料洩漏

## 常見陷阱和解決方案

### 陷阱 1：解釋器選擇錯誤
**問題**：對樹模型使用 KernelExplainer（慢且不必要）
**解決方案**：對基於樹的模型始終使用 TreeExplainer

### 陷阱 2：背景資料不足
**問題**：DeepExplainer/KernelExplainer 背景樣本太少
**解決方案**：使用 100-1000 個代表性樣本

### 陷阱 3：誤解對數賠率
**問題**：對單位的混淆（機率 vs. 對數賠率）
**解決方案**：檢查模型輸出類型；需要時使用 link="logit"

### 陷阱 4：忽略特徵相關性
**問題**：當特徵相關時將其解釋為獨立的
**解決方案**：使用特徵聚類；理解領域關係

### 陷阱 5：過度擬合解釋
**問題**：僅基於 SHAP 進行特徵工程而不驗證
**解決方案**：始終使用交叉驗證來驗證改進

### 陷阱 6：資料洩漏未檢測
**問題**：未注意到意外的特徵重要性表明洩漏
**解決方案**：根據領域知識驗證 SHAP 結果

### 陷阱 7：忽略計算限制
**問題**：為整個大型資料集計算 SHAP
**解決方案**：使用取樣、批次處理或子集分析

## 進階技術

### 技術 1：SHAP 互動值
捕捉成對特徵互動：
```python
explainer = shap.TreeExplainer(model)
shap_interaction_values = explainer.shap_interaction_values(X_test)

# 分析特定互動
feature1_idx = 0
feature2_idx = 3
interaction = shap_interaction_values[:, feature1_idx, feature2_idx]
print(f"Interaction strength: {np.abs(interaction).mean():.4f}")
```

### 技術 2：SHAP 結合部分依賴
將部分依賴圖與 SHAP 結合：
```python
from sklearn.inspection import partial_dependence

# SHAP 依賴
shap.plots.scatter(shap_values[:, "Feature1"])

# 部分依賴（模型無關）
pd_result = partial_dependence(model, X_test, features=["Feature1"])
plt.plot(pd_result['grid_values'][0], pd_result['average'][0])
```

### 技術 3：條件期望
分析基於其他特徵條件的 SHAP 值：
```python
# 高收入群組
high_income = X_test['Income'] > X_test['Income'].median()
shap.plots.beeswarm(shap_values[high_income])

# 低收入群組
low_income = X_test['Income'] <= X_test['Income'].median()
shap.plots.beeswarm(shap_values[low_income])
```

### 技術 4：冗餘性的特徵聚類
```python
# 建立階層聚類
clustering = shap.utils.hclust(X_train, y_train)

# 使用聚類視覺化
shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.5)

# 識別要移除的冗餘特徵
# 距離 < 0.1 的特徵高度冗餘
```

## 與 MLOps 的整合

**實驗追蹤**：
```python
import mlflow

# 記錄 SHAP 值
with mlflow.start_run():
    # 訓練模型
    model = train_model(X_train, y_train)

    # 計算 SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 記錄繪圖
    shap.plots.beeswarm(shap_values, show=False)
    mlflow.log_figure(plt.gcf(), "shap_beeswarm.png")
    plt.close()

    # 將特徵重要性記錄為指標
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    for feature, importance in zip(X_test.columns, mean_abs_shap):
        mlflow.log_metric(f"shap_{feature}", importance)
```

**模型監控**：
```python
# 追蹤 SHAP 分佈隨時間的漂移
def compute_shap_summary(shap_values):
    return {
        'mean': shap_values.values.mean(axis=0),
        'std': shap_values.values.std(axis=0),
        'percentiles': np.percentile(shap_values.values, [25, 50, 75], axis=0)
    }

# 計算基準
baseline_summary = compute_shap_summary(shap_values_train)

# 監控生產資料
production_summary = compute_shap_summary(shap_values_production)

# 檢測漂移
drift_detected = np.abs(
    production_summary['mean'] - baseline_summary['mean']
) > threshold
```

這份完整的工作流程文件涵蓋了實踐中 SHAP 最常見和進階的用例。
