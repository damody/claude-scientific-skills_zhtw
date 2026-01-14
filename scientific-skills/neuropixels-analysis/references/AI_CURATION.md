# AI 輔助篩選參考

使用 AI 視覺分析進行單元篩選的指南，靈感來自 SpikeAgent 的方法。

## 概述

AI 輔助篩選使用視覺語言模型分析尖峰分選視覺化，
提供類似人類篩選者的專家級品質評估。

### 工作流程

```
傳統方式：  指標 → 閾值 → 標籤
AI 增強：   指標 → AI 視覺分析 → 信心分數 → 標籤
```

## Claude Code 整合

在 Claude Code 中使用此技能時，Claude 可以直接分析波形圖，無需 API 設定。只需：

1. 產生單元報告或圖表
2. 請 Claude 分析視覺化
3. Claude 將提供專家級篩選決策

Claude Code 中的範例工作流程：
```python
# 為單元產生圖表
npa.plot_unit_summary(analyzer, unit_id=0, output='unit_0_summary.png')

# 然後詢問 Claude：「請分析此單元的波形和自相關圖
# 以判斷它是否為分離良好的單一單元、多單元活動或雜訊」
```

Claude 可以評估：
- 波形一致性和形狀
- 從自相關圖判斷反應期違規
- 振幅隨時間的穩定性
- 整體單元分離品質

## 快速開始

### 產生單元報告

```python
import neuropixels_analysis as npa

# 為單元建立視覺報告
report = npa.generate_unit_report(analyzer, unit_id=0, output_dir='reports/')

# 報告包含：
# - 波形、範本、自相關圖
# - 振幅隨時間變化、ISI 直方圖
# - 品質指標摘要
# - 用於 API 的 Base64 編碼圖片
```

### AI 視覺分析

```python
from anthropic import Anthropic

# 設定 API 客戶端
client = Anthropic()

# 分析單一單元
result = npa.analyze_unit_visually(
    analyzer,
    unit_id=0,
    api_client=client,
    model='claude-3-5-sonnet-20241022',
    task='quality_assessment'
)

print(f"Classification: {result['classification']}")
print(f"Reasoning: {result['reasoning']}")
```

### 批次分析

```python
# 分析所有單元
results = npa.batch_visual_curation(
    analyzer,
    api_client=client,
    output_dir='ai_curation/',
    progress_callback=lambda i, n: print(f"Progress: {i}/{n}")
)

# 取得標籤
ai_labels = {uid: r['classification'] for uid, r in results.items()}
```

## 互動式篩選會話

用於人機協作的 AI 輔助篩選：

```python
# 建立會話
session = npa.CurationSession.create(
    analyzer,
    output_dir='curation_session/',
    sort_by_confidence=True  # 先顯示不確定的單元
)

# 處理單元
while True:
    unit = session.current_unit()
    if unit is None:
        break

    print(f"Unit {unit.unit_id}:")
    print(f"  Auto: {unit.auto_classification} (conf: {unit.confidence:.2f})")

    # 產生報告
    report = npa.generate_unit_report(analyzer, unit.unit_id)

    # 取得 AI 意見
    ai_result = npa.analyze_unit_visually(analyzer, unit.unit_id, api_client=client)
    session.set_ai_classification(unit.unit_id, ai_result['classification'])

    # 人工決策
    decision = input("Decision (good/mua/noise/skip): ")
    if decision != 'skip':
        session.set_decision(unit.unit_id, decision)

    session.next_unit()

# 匯出結果
labels = session.get_final_labels()
session.export_decisions('final_curation.csv')
```

## 分析任務

### 品質評估（預設）

分析波形形狀、反應期、振幅穩定性。

```python
result = npa.analyze_unit_visually(analyzer, uid, task='quality_assessment')
# 回傳：'good'、'mua' 或 'noise'
```

### 合併候選偵測

判斷兩個單元是否應該合併。

```python
result = npa.analyze_unit_visually(analyzer, uid, task='merge_candidate')
# 回傳：'merge' 或 'keep_separate'
```

### 漂移評估

評估記錄中的運動/漂移。

```python
result = npa.analyze_unit_visually(analyzer, uid, task='drift_assessment')
# 回傳漂移幅度和修正建議
```

## 自訂提示

建立自訂分析提示：

```python
from neuropixels_analysis.ai_curation import create_curation_prompt

# 取得基本提示
prompt = create_curation_prompt(
    task='quality_assessment',
    additional_context='Focus on waveform amplitude consistency'
)

# 或完全自訂
custom_prompt = """
分析此單元並判斷它是否代表快速放電中間神經元。

尋找：
1. 窄波形（峰谷時間 < 0.5ms）
2. 高放電率
3. 規則的 ISI 分布

分類為：FSI（快速放電中間神經元）或 OTHER
"""

result = npa.analyze_unit_visually(
    analyzer, uid,
    api_client=client,
    custom_prompt=custom_prompt
)
```

## 結合 AI 與指標

最佳實踐：同時使用 AI 和量化指標：

```python
def hybrid_curation(analyzer, metrics, api_client):
    """結合指標和 AI 進行穩健篩選。"""
    labels = {}

    for unit_id in metrics.index:
        row = metrics.loc[unit_id]

        # 僅憑指標就具有高信心
        if row['snr'] > 10 and row['isi_violations_ratio'] < 0.001:
            labels[unit_id] = 'good'
            continue

        if row['snr'] < 1.5:
            labels[unit_id] = 'noise'
            continue

        # 不確定案例：使用 AI
        result = npa.analyze_unit_visually(
            analyzer, unit_id, api_client=api_client
        )
        labels[unit_id] = result['classification']

    return labels
```

## 會話管理

### 繼續會話

```python
# 繼續中斷的會話
session = npa.CurationSession.load('curation_session/20250101_120000/')

# 檢查進度
summary = session.get_summary()
print(f"Progress: {summary['progress_pct']:.1f}%")
print(f"Remaining: {summary['remaining']} units")

# 從中斷處繼續
unit = session.current_unit()
```

### 導覽會話

```python
# 前往特定單元
session.go_to_unit(42)

# 上一個/下一個
session.prev_unit()
session.next_unit()

# 更新決策
session.set_decision(42, 'good', notes='Clear refractory period')
```

### 匯出結果

```python
# 取得最終標籤（優先順序：人工 > AI > 自動）
labels = session.get_final_labels()

# 匯出詳細結果
df = session.export_decisions('curation_results.csv')

# 摘要
summary = session.get_summary()
print(f"Good: {summary['decisions'].get('good', 0)}")
print(f"MUA: {summary['decisions'].get('mua', 0)}")
print(f"Noise: {summary['decisions'].get('noise', 0)}")
```

## 視覺報告組件

產生的報告包含 6 個面板：

| 面板 | 內容 | 觀察重點 |
|-------|---------|------------------|
| 波形 | 個別尖峰波形 | 一致性、形狀 |
| 範本 | 平均值 ± 標準差 | 清晰的負峰、生理性形狀 |
| 自相關圖 | 尖峰時序 | 0ms 處的間隙（反應期） |
| 振幅 | 振幅隨時間變化 | 穩定性、無漂移 |
| ISI 直方圖 | 尖峰間隔 | 反應期間隙 < 1.5ms |
| 指標 | 品質數值 | SNR、ISI 違規、存在比率 |

## API 支援

目前支援的 API：

| 提供者 | 客戶端 | 模型範例 |
|----------|--------|----------------|
| Anthropic | `anthropic.Anthropic()` | claude-3-5-sonnet-20241022 |
| OpenAI | `openai.OpenAI()` | gpt-4-vision-preview |
| Google | `google.generativeai` | gemini-pro-vision |

### Anthropic 範例

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")
result = npa.analyze_unit_visually(analyzer, uid, api_client=client)
```

### OpenAI 範例

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")
result = npa.analyze_unit_visually(
    analyzer, uid,
    api_client=client,
    model='gpt-4-vision-preview'
)
```

## 最佳實踐

1. **對不確定案例使用 AI** - 不要在明顯的 good/noise 單元上浪費 API 呼叫
2. **結合指標** - AI 應該補充而非取代量化測量
3. **人工監督** - 審查 AI 決策，特別是重要分析
4. **儲存會話** - 始終使用 CurationSession 追蹤決策
5. **記錄推理** - 使用備註欄位記錄決策理由

## 成本優化

```python
# 僅對不確定的單元使用 AI
uncertain_units = metrics.query("""
    snr > 2 and snr < 8 and
    isi_violations_ratio > 0.001 and isi_violations_ratio < 0.1
""").index.tolist()

# 僅批次處理這些
results = npa.batch_visual_curation(
    analyzer,
    unit_ids=uncertain_units,
    api_client=client
)
```

## 參考資料

- [SpikeAgent](https://github.com/SpikeAgent/SpikeAgent) - AI 驅動的尖峰分選助手
- [Anthropic Vision API](https://docs.anthropic.com/en/docs/vision)
- [GPT-4 Vision](https://platform.openai.com/docs/guides/vision)
