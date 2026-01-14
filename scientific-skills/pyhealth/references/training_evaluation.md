# PyHealth 訓練、評估與可解釋性

## 概述

PyHealth 提供全面的工具，用於訓練模型、評估預測、確保模型可靠性，以及為臨床應用解釋結果。

## Trainer 類別

### 核心功能

`Trainer` 類別管理完整的模型訓練與評估工作流程，整合 PyTorch。

**初始化：**
```python
from pyhealth.trainer import Trainer

trainer = Trainer(
    model=model,  # PyHealth 或 PyTorch 模型
    device="cuda",  # 或 "cpu"
)
```

### 訓練

**train() 方法**

訓練模型，具有全面的監控與檢查點功能。

**參數：**
- `train_dataloader`：訓練資料載入器
- `val_dataloader`：驗證資料載入器（可選）
- `test_dataloader`：測試資料載入器（可選）
- `epochs`：訓練 epoch 數
- `optimizer`：優化器實例或類別
- `learning_rate`：學習率（預設：1e-3）
- `weight_decay`：L2 正則化（預設：0）
- `max_grad_norm`：梯度裁剪閾值
- `monitor`：要監控的指標（例如 "pr_auc_score"）
- `monitor_criterion`："max" 或 "min"
- `save_path`：檢查點儲存目錄

**使用方式：**
```python
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
    epochs=50,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    weight_decay=1e-5,
    max_grad_norm=5.0,
    monitor="pr_auc_score",
    monitor_criterion="max",
    save_path="./checkpoints"
)
```

**訓練功能：**

1. **自動檢查點**：根據監控指標儲存最佳模型

2. **早停**：如無改善則停止訓練

3. **梯度裁剪**：防止梯度爆炸

4. **進度追蹤**：顯示訓練進度與指標

5. **多 GPU 支援**：自動裝置配置

### 推論

**inference() 方法**

對資料集進行預測。

**參數：**
- `dataloader`：用於推論的資料載入器
- `additional_outputs`：要回傳的額外輸出列表
- `return_patient_ids`：回傳病人識別碼

**使用方式：**
```python
predictions = trainer.inference(
    dataloader=test_loader,
    additional_outputs=["attention_weights", "embeddings"],
    return_patient_ids=True
)
```

**回傳：**
- `y_pred`：模型預測
- `y_true`：真實標籤
- `patient_ids`：病人識別碼（如有請求）
- 額外輸出（如有指定）

### 評估

**evaluate() 方法**

計算全面的評估指標。

**參數：**
- `dataloader`：用於評估的資料載入器
- `metrics`：指標函數列表

**使用方式：**
```python
from pyhealth.metrics import binary_metrics_fn

results = trainer.evaluate(
    dataloader=test_loader,
    metrics=["accuracy", "pr_auc_score", "roc_auc_score", "f1_score"]
)

print(results)
# 輸出：{'accuracy': 0.85, 'pr_auc_score': 0.78, 'roc_auc_score': 0.82, 'f1_score': 0.73}
```

### 檢查點管理

**save() 方法**
```python
trainer.save("./models/best_model.pt")
```

**load() 方法**
```python
trainer.load("./models/best_model.pt")
```

## 評估指標

### 二元分類指標

**可用指標：**
- `accuracy`：整體準確率
- `precision`：精確率/陽性預測值
- `recall`：召回率/敏感度/真陽性率
- `f1_score`：F1 分數（精確率與召回率的調和平均）
- `roc_auc_score`：ROC 曲線下面積
- `pr_auc_score`：精確率-召回率曲線下面積
- `cohen_kappa`：評估者間信度

**使用方式：**
```python
from pyhealth.metrics import binary_metrics_fn

# 全面的二元指標
metrics = binary_metrics_fn(
    y_true=labels,
    y_pred=predictions,
    metrics=["accuracy", "f1_score", "pr_auc_score", "roc_auc_score"]
)
```

**閾值選擇：**
```python
# 預設閾值：0.5
predictions_binary = (predictions > 0.5).astype(int)

# 基於 F1 的最佳閾值
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

**最佳實務：**
- **使用 AUROC**：整體模型區辨力
- **使用 AUPRC**：特別是對於不平衡類別
- **使用 F1**：平衡精確率與召回率
- **報告信賴區間**：Bootstrap 重抽樣

### 多類別分類指標

**可用指標：**
- `accuracy`：整體準確率
- `macro_f1`：各類別 F1 的未加權平均
- `micro_f1`：全域 F1（總 TP、FP、FN）
- `weighted_f1`：依類別頻率加權的 F1 平均
- `cohen_kappa`：多類別 kappa

**使用方式：**
```python
from pyhealth.metrics import multiclass_metrics_fn

metrics = multiclass_metrics_fn(
    y_true=labels,
    y_pred=predictions,
    metrics=["accuracy", "macro_f1", "weighted_f1"]
)
```

**各類別指標：**
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred,
    target_names=["清醒", "N1", "N2", "N3", "REM"]))
```

**混淆矩陣：**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

### 多標籤分類指標

**可用指標：**
- `jaccard_score`：交集除以聯集
- `hamming_loss`：錯誤標籤比例
- `example_f1`：每個樣本的 F1（micro 平均）
- `label_f1`：每個標籤的 F1（macro 平均）

**使用方式：**
```python
from pyhealth.metrics import multilabel_metrics_fn

# y_pred：[n_samples, n_labels] 二元矩陣
metrics = multilabel_metrics_fn(
    y_true=label_matrix,
    y_pred=pred_matrix,
    metrics=["jaccard_score", "example_f1", "label_f1"]
)
```

**藥物推薦指標：**
```python
# Jaccard 相似度（交集/聯集）
jaccard = len(set(true_drugs) & set(pred_drugs)) / len(set(true_drugs) | set(pred_drugs))

# Precision@k：前 k 個預測的精確率
def precision_at_k(y_true, y_pred, k=10):
    top_k_pred = y_pred.argsort()[-k:]
    return len(set(y_true) & set(top_k_pred)) / k
```

### 迴歸指標

**可用指標：**
- `mean_absolute_error`：平均絕對誤差
- `mean_squared_error`：平均平方誤差
- `root_mean_squared_error`：均方根誤差
- `r2_score`：決定係數

**使用方式：**
```python
from pyhealth.metrics import regression_metrics_fn

metrics = regression_metrics_fn(
    y_true=true_values,
    y_pred=predictions,
    metrics=["mae", "rmse", "r2"]
)
```

**百分比誤差指標：**
```python
# 平均絕對百分比誤差
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 中位數絕對百分比誤差（對離群值魯棒）
medape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
```

### 公平性指標

**用途：** 評估模型在不同人口群組間的偏差

**可用指標：**
- `demographic_parity`：相等的陽性預測率
- `equalized_odds`：各群組相等的 TPR 和 FPR
- `equal_opportunity`：各群組相等的 TPR
- `predictive_parity`：各群組相等的 PPV

**使用方式：**
```python
from pyhealth.metrics import fairness_metrics_fn

fairness_results = fairness_metrics_fn(
    y_true=labels,
    y_pred=predictions,
    sensitive_attributes=demographics,  # 例如種族、性別
    metrics=["demographic_parity", "equalized_odds"]
)
```

**範例：**
```python
# 評估性別間的公平性
male_mask = (demographics == "male")
female_mask = (demographics == "female")

male_tpr = recall_score(y_true[male_mask], y_pred[male_mask])
female_tpr = recall_score(y_true[female_mask], y_pred[female_mask])

tpr_disparity = abs(male_tpr - female_tpr)
print(f"TPR 差異：{tpr_disparity:.3f}")
```

## 校正與不確定性量化

### 模型校正

**用途：** 確保預測機率與實際頻率相符

**校正圖：**
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10
)

plt.plot(mean_predicted_value, fraction_of_positives, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', label='完美校正')
plt.xlabel('平均預測機率')
plt.ylabel('正例比例')
plt.legend()
```

**期望校正誤差（ECE）：**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """計算 ECE"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1

    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_accuracy - bin_confidence)

    return ece
```

**校正方法：**

1. **Platt Scaling**：對驗證預測進行邏輯迴歸
```python
from sklearn.linear_model import LogisticRegression

calibrator = LogisticRegression()
calibrator.fit(val_predictions.reshape(-1, 1), val_labels)
calibrated_probs = calibrator.predict_proba(test_predictions.reshape(-1, 1))[:, 1]
```

2. **Isotonic Regression**：非參數校正
```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_predictions, val_labels)
calibrated_probs = calibrator.predict(test_predictions)
```

3. **Temperature Scaling**：在 softmax 前縮放 logits
```python
def find_temperature(logits, labels):
    """找到最佳溫度參數"""
    from scipy.optimize import minimize

    def nll(temp):
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=1)
        return F.cross_entropy(probs, labels).item()

    result = minimize(nll, x0=1.0, method='BFGS')
    return result.x[0]

temperature = find_temperature(val_logits, val_labels)
calibrated_logits = test_logits / temperature
```

### 不確定性量化

**Conformal Prediction：**

提供具有保證覆蓋率的預測集合。

**使用方式：**
```python
from pyhealth.metrics import prediction_set_metrics_fn

# 在驗證集上校正
scores = 1 - val_predictions[np.arange(len(val_labels)), val_labels]
quantile_level = np.quantile(scores, 0.9)  # 90% 覆蓋率

# 在測試集上生成預測集合
prediction_sets = test_predictions > (1 - quantile_level)

# 評估
metrics = prediction_set_metrics_fn(
    y_true=test_labels,
    prediction_sets=prediction_sets,
    metrics=["coverage", "average_size"]
)
```

**Monte Carlo Dropout：**

透過推論時的 dropout 估計不確定性。

```python
def predict_with_uncertainty(model, dataloader, num_samples=20):
    """使用 MC dropout 進行帶不確定性的預測"""
    model.train()  # 保持 dropout 啟用

    predictions = []
    for _ in range(num_samples):
        batch_preds = []
        for batch in dataloader:
            with torch.no_grad():
                output = model(batch)
                batch_preds.append(output)
        predictions.append(torch.cat(batch_preds))

    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)  # 不確定性

    return mean_pred, std_pred
```

**Ensemble 不確定性：**

```python
# 訓練多個模型
models = [train_model(seed=i) for i in range(5)]

# 使用 ensemble 預測
ensemble_preds = []
for model in models:
    pred = model.predict(test_data)
    ensemble_preds.append(pred)

mean_pred = np.mean(ensemble_preds, axis=0)
std_pred = np.std(ensemble_preds, axis=0)  # 不確定性
```

## 可解釋性

### 注意力視覺化

**用於 Transformer 和 RETAIN 模型：**

```python
# 在推論時取得注意力權重
outputs = trainer.inference(
    test_loader,
    additional_outputs=["attention_weights"]
)

attention = outputs["attention_weights"]

# 視覺化樣本的注意力
import matplotlib.pyplot as plt
import seaborn as sns

sample_idx = 0
sample_attention = attention[sample_idx]  # [seq_length, seq_length]

sns.heatmap(sample_attention, cmap='viridis')
plt.xlabel('Key 位置')
plt.ylabel('Query 位置')
plt.title('注意力權重')
plt.show()
```

**RETAIN 解釋：**

```python
# RETAIN 提供就診層級和特徵層級的注意力
visit_attention = outputs["visit_attention"]  # 哪些就診重要
feature_attention = outputs["feature_attention"]  # 哪些特徵重要

# 找到最有影響力的就診
most_important_visit = visit_attention[sample_idx].argmax()

# 找到該就診中最有影響力的特徵
important_features = feature_attention[sample_idx, most_important_visit].argsort()[-10:]
```

### 特徵重要性

**Permutation Importance：**

```python
from sklearn.inspection import permutation_importance

def get_predictions(model, X):
    return model.predict(X)

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    scoring='roc_auc'
)

# 按重要性排序特徵
indices = result.importances_mean.argsort()[::-1]
for i in indices[:10]:
    print(f"{feature_names[i]}: {result.importances_mean[i]:.3f}")
```

**SHAP 值：**

```python
import shap

# 建立解釋器
explainer = shap.DeepExplainer(model, train_data)

# 計算 SHAP 值
shap_values = explainer.shap_values(test_data)

# 視覺化
shap.summary_plot(shap_values, test_data, feature_names=feature_names)
```

### ChEFER（臨床健康事件特徵提取與排序）

**PyHealth 的可解釋性工具：**

```python
from pyhealth.explain import ChEFER

explainer = ChEFER(model=model, dataset=test_dataset)

# 取得預測的特徵重要性
importance_scores = explainer.explain(
    patient_id="patient_123",
    visit_id="visit_456"
)

# 視覺化最重要的特徵
explainer.plot_importance(importance_scores, top_k=20)
```

## 完整訓練管線範例

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
from pyhealth.metrics import binary_metrics_fn

# 1. 載入並準備資料
dataset = MIMIC4Dataset(root="/path/to/mimic4")
sample_dataset = dataset.set_task(mortality_prediction_mimic4_fn)

# 2. 分割資料
train_data, val_data, test_data = split_by_patient(
    sample_dataset, ratios=[0.7, 0.1, 0.2], seed=42
)

# 3. 建立資料載入器
train_loader = get_dataloader(train_data, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_data, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_data, batch_size=64, shuffle=False)

# 4. 初始化模型
model = Transformer(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "procedures", "medications"],
    mode="binary",
    embedding_dim=128,
    num_heads=8,
    num_layers=3,
    dropout=0.3
)

# 5. 訓練模型
trainer = Trainer(model=model, device="cuda")
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=50,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    weight_decay=1e-5,
    monitor="pr_auc_score",
    monitor_criterion="max",
    save_path="./checkpoints/mortality_model"
)

# 6. 在測試集上評估
test_results = trainer.evaluate(
    test_loader,
    metrics=["accuracy", "precision", "recall", "f1_score",
             "roc_auc_score", "pr_auc_score"]
)

print("測試結果：")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# 7. 取得預測進行分析
predictions = trainer.inference(test_loader, return_patient_ids=True)
y_pred, y_true, patient_ids = predictions

# 8. 校正分析
from sklearn.calibration import calibration_curve

fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
ece = expected_calibration_error(y_true, y_pred)
print(f"期望校正誤差：{ece:.4f}")

# 9. 儲存最終模型
trainer.save("./models/mortality_transformer_final.pt")
```

## 最佳實務

### 訓練

1. **監控多個指標**：追蹤損失和任務特定指標
2. **使用驗證集**：透過早停防止過擬合
3. **梯度裁剪**：穩定訓練（max_grad_norm=5.0）
4. **學習率調度**：在平原時降低學習率
5. **檢查點最佳模型**：根據驗證效能儲存

### 評估

1. **使用適合任務的指標**：二元用 AUROC/AUPRC，不平衡多類別用 macro-F1
2. **報告信賴區間**：Bootstrap 或交叉驗證
3. **分層評估**：報告各子群組的指標
4. **臨床指標**：包含臨床相關的閾值
5. **公平性評估**：評估跨人口群組的表現

### 部署

1. **校正預測**：確保機率可靠
2. **量化不確定性**：提供信心估計
3. **監控效能**：追蹤生產環境中的指標
4. **處理分佈漂移**：檢測資料何時改變
5. **可解釋性**：為預測提供解釋
