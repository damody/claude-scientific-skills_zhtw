# USPTO 商標 API 參考

## 概述

USPTO 提供兩個主要的商標資料 API：

1. **商標狀態與文件檢索（TSDR）** - 檢索商標案件狀態和文件
2. **商標轉讓搜尋** - 搜尋商標轉讓記錄

## 1. 商標狀態與文件檢索（TSDR）API

### 概述

TSDR 支援程式化檢索商標案件狀態文件和資訊。

**API 版本：** v1.0

**基礎 URL：** `https://tsdrapi.uspto.gov/ts/cd/`

### 認證

需要在以下網址註冊 API 金鑰：https://account.uspto.gov/api-manager/

在請求標頭中包含 API 金鑰：
```
X-Api-Key: YOUR_API_KEY
```

### 端點

#### 按序號取得商標狀態

```
GET /ts/cd/casedocs/sn{serial_number}/info.json
```

**範例：**
```bash
curl -H "X-Api-Key: YOUR_KEY" \
  "https://tsdrapi.uspto.gov/ts/cd/casedocs/sn87654321/info.json"
```

#### 按註冊號取得商標狀態

```
GET /ts/cd/casedocs/rn{registration_number}/info.json
```

### 回應格式

回傳包含完整商標資訊的 JSON：

```json
{
  "TradeMarkAppln": {
    "ApplicationNumber": "87654321",
    "ApplicationDate": "2017-10-15",
    "RegistrationNumber": "5678901",
    "RegistrationDate": "2019-03-12",
    "MarkVerbalElementText": "EXAMPLE MARK",
    "MarkCurrentStatusExternalDescriptionText": "REGISTERED",
    "MarkCurrentStatusDate": "2019-03-12",
    "GoodsAndServices": [...],
    "Owners": [...],
    "Correspondents": [...]
  }
}
```

### 關鍵資料欄位

- **申請資訊：**
  - `ApplicationNumber` - 序號
  - `ApplicationDate` - 申請日期
  - `ApplicationType` - 類型（TEAS Plus、TEAS Standard 等）

- **註冊資訊：**
  - `RegistrationNumber` - 註冊號（如已註冊）
  - `RegistrationDate` - 註冊日期

- **商標資訊：**
  - `MarkVerbalElementText` - 商標文字
  - `MarkCurrentStatusExternalDescriptionText` - 目前狀態
  - `MarkCurrentStatusDate` - 狀態日期
  - `MarkDrawingCode` - 商標類型（文字、設計等）

- **分類：**
  - `GoodsAndServices` - 商品/服務陣列及類別

- **所有權人資訊：**
  - `Owners` - 商標所有權人/申請人陣列

- **審查歷史：**
  - `ProsecutionHistoryEntry` - 審查事件陣列

### 常見狀態值

- **REGISTERED** - 商標已註冊且有效
- **PENDING** - 申請審查中
- **ABANDONED** - 申請/註冊已放棄
- **CANCELLED** - 註冊已取消
- **SUSPENDED** - 審查暫停
- **PUBLISHED FOR OPPOSITION** - 已公告，在異議期內
- **REGISTERED AND RENEWED** - 註冊已續展

### Python 範例

```python
import requests

def get_trademark_status(serial_number, api_key):
    """按序號檢索商標狀態。"""
    url = f"https://tsdrapi.uspto.gov/ts/cd/casedocs/sn{serial_number}/info.json"
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API 錯誤：{response.status_code}")

# 使用方式
data = get_trademark_status("87654321", "YOUR_API_KEY")
trademark = data['TradeMarkAppln']

print(f"商標：{trademark['MarkVerbalElementText']}")
print(f"狀態：{trademark['MarkCurrentStatusExternalDescriptionText']}")
print(f"申請日期：{trademark['ApplicationDate']}")
if 'RegistrationNumber' in trademark:
    print(f"註冊號：{trademark['RegistrationNumber']}")
```

## 2. 商標轉讓搜尋 API

### 概述

從 USPTO 轉讓資料庫檢索商標轉讓記錄。顯示所有權轉移和擔保權益。

**API 版本：** v1.4

**基礎 URL：** `https://assignment-api.uspto.gov/trademark/`

### 認證

需要在標頭中包含 API 金鑰：
```
X-Api-Key: YOUR_API_KEY
```

### 搜尋方法

#### 按註冊號

```
GET /v1.4/assignment/application/{registration_number}
```

#### 按序號

```
GET /v1.4/assignment/application/{serial_number}
```

#### 按受讓人名稱

```
POST /v1.4/assignment/search
```

**請求主體：**
```json
{
  "criteria": {
    "assigneeName": "Company Name"
  }
}
```

### 回應格式

回傳包含轉讓記錄的 XML：

```xml
<assignments>
  <assignment>
    <reelFrame>12345/0678</reelFrame>
    <conveyanceText>ASSIGNMENT OF ASSIGNORS INTEREST</conveyanceText>
    <recordedDate>2020-01-15</recordedDate>
    <executionDate>2020-01-10</executionDate>
    <assignors>
      <assignor>
        <name>Original Owner LLC</name>
      </assignor>
    </assignors>
    <assignees>
      <assignee>
        <name>New Owner Corporation</name>
      </assignee>
    </assignees>
  </assignment>
</assignments>
```

### 關鍵欄位

- `reelFrame` - USPTO 捲軸和畫面編號
- `conveyanceText` - 交易類型
- `recordedDate` - USPTO 記錄日期
- `executionDate` - 文件執行日期
- `assignors` - 原所有權人
- `assignees` - 新所有權人
- `propertyNumbers` - 受影響的序號/註冊號

### 常見轉讓類型

- **ASSIGNMENT OF ASSIGNORS INTEREST** - 所有權轉移
- **SECURITY AGREEMENT** - 抵押/擔保權益
- **MERGER** - 公司合併
- **CHANGE OF NAME** - 名稱變更
- **ASSIGNMENT OF PARTIAL INTEREST** - 部分所有權轉移

### Python 範例

```python
import requests
import xml.etree.ElementTree as ET

def search_trademark_assignments(registration_number, api_key):
    """搜尋商標註冊的轉讓記錄。"""
    url = f"https://assignment-api.uspto.gov/trademark/v1.4/assignment/application/{registration_number}"
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text  # 回傳 XML
    else:
        raise Exception(f"API 錯誤：{response.status_code}")

# 使用方式
xml_data = search_trademark_assignments("5678901", "YOUR_API_KEY")
root = ET.fromstring(xml_data)

for assignment in root.findall('.//assignment'):
    reel_frame = assignment.find('reelFrame').text
    recorded_date = assignment.find('recordedDate').text
    conveyance = assignment.find('conveyanceText').text

    assignor = assignment.find('.//assignor/name').text
    assignee = assignment.find('.//assignee/name').text

    print(f"{recorded_date}：{assignor} -> {assignee}")
    print(f"  類型：{conveyance}")
    print(f"  捲軸/畫面：{reel_frame}\n")
```

## 使用案例

### 1. 監控商標狀態

檢查待審申請或註冊的狀態：

```python
def check_trademark_health(serial_number, api_key):
    """檢查商標是否需要注意。"""
    data = get_trademark_status(serial_number, api_key)
    tm = data['TradeMarkAppln']

    status = tm['MarkCurrentStatusExternalDescriptionText']
    alerts = []

    if 'ABANDON' in status:
        alerts.append("警告：已放棄")
    elif 'PUBLISHED' in status:
        alerts.append("通知：在異議期內")
    elif 'SUSPENDED' in status:
        alerts.append("暫停：審查暫停")
    elif 'REGISTERED' in status:
        alerts.append("正常：有效")

    return alerts
```

### 2. 追蹤所有權變更

監控轉讓記錄的所有權變更：

```python
def get_current_owner(registration_number, api_key):
    """從轉讓記錄尋找目前商標所有權人。"""
    xml_data = search_trademark_assignments(registration_number, api_key)
    root = ET.fromstring(xml_data)

    assignments = []
    for assignment in root.findall('.//assignment'):
        date = assignment.find('recordedDate').text
        assignee = assignment.find('.//assignee/name').text
        assignments.append((date, assignee))

    # 最近的轉讓
    if assignments:
        assignments.sort(reverse=True)
        return assignments[0][1]
    return None
```

### 3. 商標組合管理

分析商標組合：

```python
def analyze_portfolio(serial_numbers, api_key):
    """分析多個商標的狀態。"""
    results = {
        'active': 0,
        'pending': 0,
        'abandoned': 0,
        'expired': 0
    }

    for sn in serial_numbers:
        data = get_trademark_status(sn, api_key)
        status = data['TradeMarkAppln']['MarkCurrentStatusExternalDescriptionText']

        if 'REGISTERED' in status:
            results['active'] += 1
        elif 'PENDING' in status or 'PUBLISHED' in status:
            results['pending'] += 1
        elif 'ABANDON' in status:
            results['abandoned'] += 1
        elif 'EXPIRED' in status or 'CANCELLED' in status:
            results['expired'] += 1

    return results
```

## 速率限制和最佳實務

1. **遵守速率限制** - 實作具有指數退避的重試邏輯
2. **快取回應** - 商標資料不常變更
3. **批次處理** - 對大型組合分散請求時間
4. **錯誤處理** - 優雅處理缺失資料（並非所有商標都有所有欄位）
5. **資料驗證** - 在 API 呼叫前驗證序號/註冊號格式

## 與其他資料整合

結合商標資料與其他來源：

- **TSDR + 轉讓** - 目前狀態 + 所有權歷史
- **多個商標** - 分析家族中的相關商標
- **專利資料** - 交叉參考智慧財產權組合

## 資源

- **TSDR API**：https://developer.uspto.gov/api-catalog/tsdr-data-api
- **轉讓 API**：https://developer.uspto.gov/api-catalog/trademark-assignment-search-data-api
- **API 金鑰註冊**：https://account.uspto.gov/api-manager/
- **商標搜尋**：https://tmsearch.uspto.gov/
- **Swagger 文件**：https://developer.uspto.gov/swagger/tsdr-api-v1
