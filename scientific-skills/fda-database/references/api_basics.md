# OpenFDA API 基礎

本參考文件提供使用 openFDA API 的完整資訊，包括認證、速率限制、查詢語法和最佳實踐。

## 開始使用

### 基礎 URL

所有 openFDA API 端點遵循此結構：
```
https://api.fda.gov/{category}/{endpoint}.json
```

範例：
- `https://api.fda.gov/drug/event.json`
- `https://api.fda.gov/device/510k.json`
- `https://api.fda.gov/food/enforcement.json`

### 必須使用 HTTPS

**所有請求必須使用 HTTPS**。HTTP 請求不被接受且會失敗。

## 認證

### API 金鑰註冊

雖然 openFDA 可以在沒有 API 金鑰的情況下使用，但強烈建議註冊免費的 API 金鑰以獲得更高的速率限制。

**註冊**：請造訪 https://open.fda.gov/apis/authentication/ 進行註冊

**API 金鑰的好處**：
- 更高的速率限制（240 請求/分鐘，120,000 請求/天）
- 更適合生產應用程式
- 無額外費用

### 使用您的 API 金鑰

在請求中包含您的 API 金鑰可使用以下兩種方法之一：

**方法 1：查詢參數（建議）**
```python
import requests

api_key = "YOUR_API_KEY_HERE"
url = "https://api.fda.gov/drug/event.json"

params = {
    "api_key": api_key,
    "search": "patient.drug.medicinalproduct:aspirin",
    "limit": 10
}

response = requests.get(url, params=params)
```

**方法 2：基本認證**
```python
import requests

api_key = "YOUR_API_KEY_HERE"
url = "https://api.fda.gov/drug/event.json"

params = {
    "search": "patient.drug.medicinalproduct:aspirin",
    "limit": 10
}

response = requests.get(url, params=params, auth=(api_key, ''))
```

## 速率限制

### 目前限制

| 狀態 | 每分鐘請求數 | 每日請求數 |
|--------|-------------------|------------------|
| **無 API 金鑰** | 每個 IP 地址 240 個 | 每個 IP 地址 1,000 個 |
| **有 API 金鑰** | 每個金鑰 240 個 | 每個金鑰 120,000 個 |

### 速率限制標頭

API 在回應標頭中回傳速率限制資訊：
```python
response = requests.get(url, params=params)

print(f"速率限制：{response.headers.get('X-RateLimit-Limit')}")
print(f"剩餘：{response.headers.get('X-RateLimit-Remaining')}")
print(f"重置時間：{response.headers.get('X-RateLimit-Reset')}")
```

### 處理速率限制

當您超過速率限制時，API 回傳：
- **狀態碼**：`429 Too Many Requests`
- **錯誤訊息**：指示超過速率限制

**最佳實踐**：實作指數退避：
```python
import requests
import time

def query_with_rate_limit_handling(url, params, max_retries=3):
    """使用自動速率限制處理查詢 API。"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                # 超過速率限制
                wait_time = (2 ** attempt) * 60  # 指數退避
                print(f"達到速率限制。等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("超過最大重試次數")
```

### 提高限制

對於需要更高限制的應用程式，請透過其網站聯繫 openFDA 團隊，並提供您的使用案例詳情。

## 查詢語法

### 基本結構

查詢使用此格式：
```
?api_key=YOUR_KEY&parameter=value&parameter2=value2
```

參數由 & 符號（`&`）分隔。

### Search 參數

`search` 參數是過濾結果的主要方式。

**基本格式**：
```
search=field:value
```

**範例**：
```python
params = {
    "api_key": api_key,
    "search": "patient.drug.medicinalproduct:aspirin"
}
```

### 搜尋運算子

#### AND 運算子
組合多個條件（兩者都必須為真）：
```python
# 查詢加拿大的阿斯匹靈不良事件
params = {
    "search": "patient.drug.medicinalproduct:aspirin+AND+occurcountry:ca"
}
```

#### OR 運算子
任一條件可以為真（OR 隱含於空格）：
```python
# 查詢阿斯匹靈或布洛芬
params = {
    "search": "patient.drug.medicinalproduct:(aspirin ibuprofen)"
}
```

或明確表示：
```python
params = {
    "search": "patient.drug.medicinalproduct:aspirin+OR+patient.drug.medicinalproduct:ibuprofen"
}
```

#### NOT 運算子
排除結果：
```python
# 非美國的事件
params = {
    "search": "_exists_:occurcountry+AND+NOT+occurcountry:us"
}
```

#### 萬用字元
使用星號（`*`）進行部分匹配：
```python
# 任何以「met」開頭的藥物
params = {
    "search": "patient.drug.medicinalproduct:met*"
}

# 任何包含「cillin」的藥物
params = {
    "search": "patient.drug.medicinalproduct:*cillin*"
}
```

#### 精確短語匹配
使用引號進行精確短語匹配：
```python
params = {
    "search": 'patient.reaction.reactionmeddrapt:"heart attack"'
}
```

#### 範圍查詢
在範圍內搜尋：
```python
# 日期範圍（YYYYMMDD 格式）
params = {
    "search": "receivedate:[20200101+TO+20201231]"
}

# 數字範圍
params = {
    "search": "patient.patientonsetage:[18+TO+65]"
}

# 開放式範圍
params = {
    "search": "patient.patientonsetage:[65+TO+*]"  # 65 歲及以上
}
```

#### 欄位存在性
檢查欄位是否存在：
```python
# 具有患者年齡的記錄
params = {
    "search": "_exists_:patient.patientonsetage"
}

# 缺少患者年齡的記錄
params = {
    "search": "_missing_:patient.patientonsetage"
}
```

### Limit 參數

控制回傳多少結果（1-1000，預設 1）：
```python
params = {
    "search": "...",
    "limit": 100
}
```

**最大值**：每個請求 1000 個結果

### Skip 參數

用於分頁，跳過前 N 個結果：
```python
# 取得結果 101-200
params = {
    "search": "...",
    "limit": 100,
    "skip": 100
}
```

**分頁範例**：
```python
def get_all_results(url, search_query, api_key, max_results=5000):
    """使用分頁擷取結果。"""
    all_results = []
    skip = 0
    limit = 100

    while len(all_results) < max_results:
        params = {
            "api_key": api_key,
            "search": search_query,
            "limit": limit,
            "skip": skip
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "results" not in data or len(data["results"]) == 0:
            break

        all_results.extend(data["results"])

        if len(data["results"]) < limit:
            break  # 沒有更多結果

        skip += limit
        time.sleep(0.25)  # 速率限制禮貌

    return all_results[:max_results]
```

### Count 參數

按欄位聚合和計數結果（而非回傳個別記錄）：
```python
# 按國家計數事件
params = {
    "search": "patient.drug.medicinalproduct:aspirin",
    "count": "occurcountry"
}
```

**回應格式**：
```json
{
  "results": [
    {"term": "us", "count": 12543},
    {"term": "ca", "count": 3421},
    {"term": "gb", "count": 2156}
  ]
}
```

#### 精確計數

添加 `.exact` 後綴進行精確短語計數（對於多詞欄位特別重要）：
```python
# 計數精確反應術語（不是個別詞）
params = {
    "search": "patient.drug.medicinalproduct:aspirin",
    "count": "patient.reaction.reactionmeddrapt.exact"
}
```

**無 `.exact`**：計數個別詞
**有 `.exact`**：計數完整短語

### Sort 參數

按欄位排序結果：
```python
# 按日期排序，最新在前
params = {
    "search": "...",
    "sort": "receivedate:desc"
}

# 按日期排序，最舊在前
params = {
    "search": "...",
    "sort": "receivedate:asc"
}
```

## 回應格式

### 標準回應結構

```json
{
  "meta": {
    "disclaimer": "...",
    "terms": "...",
    "license": "...",
    "last_updated": "2024-01-15",
    "results": {
      "skip": 0,
      "limit": 10,
      "total": 15234
    }
  },
  "results": [
    {
      // 個別結果記錄
    },
    {
      // 另一個結果記錄
    }
  ]
}
```

### 回應欄位

- **meta**：關於查詢和結果的元資料
  - `disclaimer`：重要法律免責聲明
  - `terms`：使用條款 URL
  - `license`：資料授權資訊
  - `last_updated`：資料最後更新時間
  - `results.skip`：跳過的結果數
  - `results.limit`：每頁最大結果數
  - `results.total`：總匹配結果數（對於大型結果集可能是近似值）

- **results**：匹配記錄的陣列

### 空結果

當沒有結果匹配時：
```json
{
  "meta": {...},
  "results": []
}
```

### 錯誤回應

當發生錯誤時：
```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "詳細錯誤訊息"
  }
}
```

**常見錯誤代碼**：
- `NOT_FOUND`：未找到結果（404）
- `INVALID_QUERY`：格式錯誤的搜尋查詢（400）
- `RATE_LIMIT_EXCEEDED`：請求過多（429）
- `UNAUTHORIZED`：無效的 API 金鑰（401）
- `SERVER_ERROR`：內部伺服器錯誤（500）

## 進階技術

### 巢狀欄位查詢

查詢巢狀物件：
```python
# 嚴重結果為死亡的藥物不良事件
params = {
    "search": "serious:1+AND+seriousnessdeath:1"
}
```

### 多欄位搜尋

跨多個欄位搜尋：
```python
# 在多個欄位中搜尋藥物名稱
params = {
    "search": "(patient.drug.medicinalproduct:aspirin+OR+patient.drug.openfda.brand_name:aspirin)"
}
```

### 複雜布林邏輯

組合多個運算子：
```python
# (阿斯匹靈 OR 布洛芬) AND (心臟病發) AND NOT (美國)
params = {
    "search": "(patient.drug.medicinalproduct:aspirin+OR+patient.drug.medicinalproduct:ibuprofen)+AND+patient.reaction.reactionmeddrapt:*heart*attack*+AND+NOT+occurcountry:us"
}
```

### 帶過濾器的計數

在特定子集中計數：
```python
# 僅計數嚴重事件的反應
params = {
    "search": "serious:1",
    "count": "patient.reaction.reactionmeddrapt.exact"
}
```

## 最佳實踐

### 1. 查詢效率

**建議**：
- 使用特定欄位搜尋
- 在計數前過濾
- 盡可能使用精確匹配
- 對大型資料集實作分頁

**避免**：
- 使用過於廣泛的萬用字元（例如 `search=*`）
- 請求超過需要的資料
- 跳過錯誤處理
- 忽略速率限制

### 2. 錯誤處理

始終處理常見錯誤：
```python
def safe_api_call(url, params):
    """安全地呼叫 FDA API，具有完整的錯誤處理。"""
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"error": "未找到結果"}
        elif response.status_code == 429:
            return {"error": "超過速率限制"}
        elif response.status_code == 400:
            return {"error": "無效查詢"}
        else:
            return {"error": f"HTTP 錯誤：{e}"}
    except requests.exceptions.ConnectionError:
        return {"error": "連線失敗"}
    except requests.exceptions.Timeout:
        return {"error": "請求逾時"}
    except requests.exceptions.RequestException as e:
        return {"error": f"請求錯誤：{e}"}
```

### 3. 資料驗證

驗證和清理資料：
```python
def clean_search_term(term):
    """清理和準備搜尋詞。"""
    # 移除會破壞查詢的特殊字元
    term = term.replace('"', '\\"')  # 跳脫引號
    term = term.strip()
    return term

def validate_date(date_str):
    """驗證日期格式（YYYYMMDD）。"""
    import re
    if not re.match(r'^\d{8}$', date_str):
        raise ValueError("日期必須為 YYYYMMDD 格式")
    return date_str
```

### 4. 快取

對頻繁存取的資料實作快取：
```python
import json
from pathlib import Path
import hashlib
import time

class FDACache:
    """FDA API 回應的簡單檔案快取。"""

    def __init__(self, cache_dir="fda_cache", ttl=3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl  # 存活時間（秒）

    def _get_cache_key(self, url, params):
        """從 URL 和參數產生快取金鑰。"""
        cache_string = f"{url}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, url, params):
        """如果可用且未過期則取得快取回應。"""
        key = self._get_cache_key(url, params)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            # 檢查是否過期
            age = time.time() - cache_file.stat().st_mtime
            if age < self.ttl:
                with open(cache_file, 'r') as f:
                    return json.load(f)

        return None

    def set(self, url, params, data):
        """快取回應資料。"""
        key = self._get_cache_key(url, params)
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w') as f:
            json.dump(data, f)

# 使用方式
cache = FDACache(ttl=3600)  # 1 小時快取

def cached_api_call(url, params):
    """帶快取的 API 呼叫。"""
    # 檢查快取
    cached = cache.get(url, params)
    if cached:
        return cached

    # 發出請求
    response = requests.get(url, params=params)
    data = response.json()

    # 快取結果
    cache.set(url, params, data)

    return data
```

### 5. 速率限制管理

追蹤和遵守速率限制：
```python
import time
from collections import deque

class RateLimiter:
    """追蹤和強制速率限制。"""

    def __init__(self, max_per_minute=240):
        self.max_per_minute = max_per_minute
        self.requests = deque()

    def wait_if_needed(self):
        """如有必要則等待以保持在速率限制內。"""
        now = time.time()

        # 移除超過 1 分鐘的請求
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()

        # 檢查是否達到限制
        if len(self.requests) >= self.max_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.requests.popleft()

        self.requests.append(time.time())

# 使用方式
rate_limiter = RateLimiter(max_per_minute=240)

def rate_limited_request(url, params):
    """帶速率限制的請求。"""
    rate_limiter.wait_if_needed()
    return requests.get(url, params=params)
```

## 常見查詢模式

### 模式 1：基於時間的分析
```python
# 取得過去 30 天的事件
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

params = {
    "search": f"receivedate:[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]",
    "limit": 1000
}
```

### 模式 2：前 N 名分析
```python
# 取得某藥物最常見的 10 個反應
params = {
    "search": "patient.drug.medicinalproduct:aspirin",
    "count": "patient.reaction.reactionmeddrapt.exact",
    "limit": 10
}
```

### 模式 3：比較分析
```python
# 比較兩種藥物
drugs = ["aspirin", "ibuprofen"]
results = {}

for drug in drugs:
    params = {
        "search": f"patient.drug.medicinalproduct:{drug}",
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": 10
    }
    results[drug] = requests.get(url, params=params).json()
```

## 其他資源

- **openFDA 首頁**：https://open.fda.gov/
- **API 文件**：https://open.fda.gov/apis/
- **互動式 API 探索器**：https://open.fda.gov/apis/try-the-api/
- **服務條款**：https://open.fda.gov/terms/
- **GitHub**：https://github.com/FDA/openfda
- **狀態頁面**：查看 API 中斷和維護

## 支援

如有問題或疑慮：
- **GitHub Issues**：https://github.com/FDA/openfda/issues
- **電子郵件**：open-fda@fda.hhs.gov
- **討論論壇**：查看 GitHub discussions
