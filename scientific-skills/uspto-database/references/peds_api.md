# 專利審查資料系統（PEDS）API 參考

## 概述

專利審查資料系統（PEDS）提供 USPTO 專利申請和申請狀態記錄的存取。它包含書目資料、已公開文件資訊和專利期限延長資料。

**資料涵蓋範圍：** 1981 年至今（部分資料可追溯至 1935 年）

**基礎 URL：** 透過 USPTO 開放資料入口存取

## PEDS 提供的內容

PEDS 提供專利申請的全面交易歷史和狀態資訊：

- **書目資料** - 申請號、申請日期、標題、發明人、受讓人
- **已公開文件** - 公開號和公開日期
- **交易歷史** - 所有審查事件，包含日期、代碼和描述
- **專利期限調整** - PTA/PTE 資訊
- **申請狀態** - 目前狀態和狀態代碼
- **文件包裝存取** - 審查文件連結

## 主要功能

1. **交易活動** - 包含交易日期、代碼和描述的完整審查時間線
2. **狀態資訊** - 目前申請狀態和狀態代碼
3. **書目更新** - 發明人、受讓人、標題隨時間的變更
4. **家族資料** - 相關申請和延續資料
5. **審查意見通知書追蹤** - 郵寄日期和審查意見通知書資訊

## Python 函式庫：uspto-opendata-python

存取 PEDS 的建議方式是透過 `uspto-opendata-python` 函式庫。

### 安裝

```bash
pip install uspto-opendata-python
```

### 基本用法

```python
from uspto.peds import PEDSClient

# 初始化客戶端
client = PEDSClient()

# 按申請號搜尋
app_number = "16123456"
result = client.get_application(app_number)

# 存取申請資料
print(f"標題：{result['title']}")
print(f"申請日期：{result['filing_date']}")
print(f"狀態：{result['status']}")

# 取得交易歷史
transactions = result['transactions']
for trans in transactions:
    print(f"{trans['date']}：{trans['code']} - {trans['description']}")
```

### 搜尋方法

```python
# 按申請號
client.get_application("16123456")

# 按專利號
client.get_patent("11234567")

# 按客戶號（受讓人）
client.search_by_customer_number("12345")

# 批次檢索
app_numbers = ["16123456", "16123457", "16123458"]
results = client.bulk_retrieve(app_numbers)
```

## 資料欄位

### 書目欄位

- `application_number` - 申請號
- `filing_date` - 申請日期
- `patent_number` - 專利號（如已核准）
- `patent_issue_date` - 核准日期（如已核准）
- `title` - 申請/專利標題
- `inventors` - 發明人列表
- `assignees` - 受讓人列表
- `app_type` - 申請類型（發明、設計、植物、再發行）
- `app_status` - 目前申請狀態
- `app_status_date` - 狀態日期

### 交易欄位

- `transaction_date` - 交易日期
- `transaction_code` - USPTO 事件代碼
- `transaction_description` - 事件描述
- `mail_date` - 郵件室日期（用於審查意見通知書）

### 專利期限資料

- `pta_pte_summary` - 專利期限調整/延長摘要
- `pta_pte_history` - 期限計算歷史

## 狀態代碼

常見申請狀態代碼：

- **Patented Case** - 專利已核准
- **Abandoned** - 申請已放棄
- **Pending** - 申請審查中
- **Allowed** - 申請已核准，等待發證
- **Final Rejection** - 已發出最終核駁
- **Non-Final Rejection** - 已發出非最終核駁
- **Response Filed** - 申請人已提交回覆

## 交易代碼

常見交易代碼包括：

- **CTNF** - 非最終核駁郵寄
- **CTFR** - 最終核駁郵寄
- **AOPF** - 審查意見通知書郵寄
- **WRIT** - 回覆提交
- **NOA** - 核准通知郵寄
- **ISS.FEE** - 發證費繳納
- **ABND** - 申請放棄

完整代碼列表請參閱 OCE 專利審查狀態/事件代碼 API。

## 使用案例

### 1. 追蹤申請進度

監控待審申請的審查意見通知書和狀態變更。

```python
# 取得目前狀態
app = client.get_application("16123456")
print(f"目前狀態：{app['app_status']}")
print(f"狀態日期：{app['app_status_date']}")

# 檢查最近的審查意見通知書
recent_oas = [t for t in app['transactions']
              if t['code'] in ['CTNF', 'CTFR', 'AOPF']
              and t['date'] > '2024-01-01']
```

### 2. 專利組合分析

分析整個專利組合的審查歷史。

```python
# 取得受讓人的所有申請
apps = client.search_by_customer_number("12345")

# 計算平均待審時間
pendencies = []
for app in apps:
    if app['patent_issue_date']:
        filing = datetime.strptime(app['filing_date'], '%Y-%m-%d')
        issue = datetime.strptime(app['patent_issue_date'], '%Y-%m-%d')
        pendencies.append((issue - filing).days)

avg_pendency = sum(pendencies) / len(pendencies)
print(f"平均待審時間：{avg_pendency} 天")
```

### 3. 檢查核駁模式

分析收到的核駁類型。

```python
# 統計核駁類型
rejections = {}
for trans in app['transactions']:
    if 'rejection' in trans['description'].lower():
        code = trans['code']
        rejections[code] = rejections.get(code, 0) + 1
```

## 與其他 API 整合

PEDS 資料可以與其他 USPTO API 結合：

- **審查意見通知書文字 API** - 使用申請號檢索審查意見通知書全文
- **專利轉讓搜尋** - 尋找所有權變更
- **PTAB API** - 檢查上訴程序

## 重要注意事項

1. **PAIR 批量資料（PBD）已停用** - 請改用 PEDS
2. **資料更新** - PEDS 定期更新，但可能有 1-2 天的延遲
3. **申請號** - 使用標準化格式（無斜線或空格）
4. **延續資料** - 母/子申請在交易歷史中追蹤

## 最佳實務

1. **批次請求** - 對多個申請使用批次檢索
2. **快取資料** - 避免對同一申請的重複 API 呼叫
3. **監控更新** - 定期檢查交易更新
4. **處理缺失資料** - 並非所有申請都填充所有欄位
5. **解析交易代碼** - 使用代碼描述進行使用者友善的顯示

## 資源

- **函式庫文件**：https://docs.ip-tools.org/uspto-opendata-python/
- **PyPI 套件**：https://pypi.org/project/uspto-opendata-python/
- **GitHub 儲存庫**：https://github.com/ip-tools/uspto-opendata-python
- **USPTO PEDS 入口**：https://ped.uspto.gov/
