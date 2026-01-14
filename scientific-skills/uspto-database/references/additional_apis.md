# 額外 USPTO API 參考

## 概述

除了專利搜尋、PEDS 和商標之外，USPTO 還提供專門的 API 用於引用、審查意見通知書、轉讓、訴訟和其他專利資料。

## 1. 增強引用 API

### 概述

為 IP5（USPTO、EPO、JPO、KIPO、CNIPA）和公眾使用提供專利評估過程和引用參考的洞察。

**版本：** v3、v2、v1

**基礎 URL：** 透過 USPTO 開放資料入口存取

### 目的

分析審查員在專利審查期間引用哪些參考文獻，以及專利如何引用先前技術。

### 主要功能

- **前向引用** - 引用特定專利的專利
- **後向引用** - 專利引用的參考文獻
- **審查員引用** - 審查員引用 vs. 申請人引用的參考文獻
- **引用上下文** - 如何以及為何引用參考文獻

### 使用案例

- 先前技術分析
- 專利地景分析
- 識別相關技術
- 根據引用評估專利強度

## 2. 審查意見通知書 API

### 2.1 審查意見通知書文字檢索 API

**版本：** v1

### 目的

檢索專利申請的完整審查意見通知書通信文件全文。

### 功能

- 審查意見通知書全文
- 限制、核駁、反對意見
- 審查員修改
- 檢索資訊

### 使用範例

```python
# 按申請號檢索審查意見通知書文字
def get_office_action_text(app_number, api_key):
    """
    取得申請的審查意見通知書全文。
    注意：與 PEDS 整合以識別存在哪些審查意見通知書。
    """
    # API 實作
    pass
```

### 2.2 審查意見通知書引用 API

**版本：** v2、beta v1

### 目的

提供從審查意見通知書中提取的專利引用資料，顯示審查員在審查期間使用了哪些參考文獻。

### 關鍵資料

- 專利和非專利文獻引用
- 引用上下文（核駁、資訊等）
- 審查員搜尋策略
- 審查研究資料集

### 2.3 審查意見通知書核駁 API

**版本：** v2、beta v1

### 目的

詳細說明核駁原因和審查結果，提供截至 2025 年 3 月的批量核駁資料。

### 核駁類型

- **35 U.S.C. § 102** - 預期（缺乏新穎性）
- **35 U.S.C. § 103** - 顯而易見性
- **35 U.S.C. § 112** - 可實施性、書面描述、不明確性
- **35 U.S.C. § 101** - 主題適格性

### 使用案例

- 分析常見核駁原因
- 識別有問題的請求項語言
- 根據歷史資料準備回覆
- 專利組合的核駁模式分析

### 2.4 審查意見通知書每週壓縮檔 API

**版本：** v1

### 目的

提供按每週發布時程組織的審查意見通知書文件全文批量下載。

### 功能

- 每週檔案下載
- 完整審查意見通知書文字
- 大規模分析的批量存取

## 3. 專利轉讓搜尋 API

### 概述

**版本：** v1.4

存取 USPTO 專利轉讓資料庫的所有權記錄和轉移。

**基礎 URL：** `https://assignment-api.uspto.gov/patent/`

### 目的

追蹤專利所有權、轉讓、擔保權益和公司交易。

### 搜尋方法

#### 按專利號

```
GET /v1.4/assignment/patent/{patent_number}
```

#### 按申請號

```
GET /v1.4/assignment/application/{application_number}
```

#### 按受讓人名稱

```
POST /v1.4/assignment/search
{
  "criteria": {
    "assigneeName": "Company Name"
  }
}
```

### 回應格式

回傳與商標轉讓類似的 XML 轉讓記錄：

- 捲軸/畫面編號
- 轉讓類型
- 日期（執行日和記錄日）
- 讓與人和受讓人
- 受影響的專利/申請

### 常見用途

```python
def track_patent_ownership(patent_number, api_key):
    """追蹤專利的所有權歷史。"""
    url = f"https://assignment-api.uspto.gov/patent/v1.4/assignment/patent/{patent_number}"
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # 解析 XML 以提取轉讓歷史
        return response.text
    return None

def find_company_patents(company_name, api_key):
    """尋找轉讓給公司的專利。"""
    url = "https://assignment-api.uspto.gov/patent/v1.4/assignment/search"
    headers = {"X-Api-Key": api_key}
    data = {"criteria": {"assigneeName": company_name}}

    response = requests.post(url, headers=headers, json=data)
    return response.text
```

## 4. PTAB API（專利審判和上訴委員會）

### 概述

**版本：** v2

存取專利審判和上訴委員會程序資料。

### 目的

檢索以下資訊：
- 雙方複審（IPR）
- 核准後複審（PGR）
- 涵蓋商業方法（CBM）複審
- 單方上訴

### 可用資料

- 請願資訊
- 審判決定
- 最終書面決定
- 請願人和專利權人資訊
- 被質疑的請求項
- 審判結果

### 注意

目前正在遷移到新的開放資料入口。請查閱目前文件以獲取存取詳情。

## 5. 專利訴訟案件 API

### 概述

**版本：** v1

包含 74,623+ 地區法院訴訟記錄，涵蓋專利訴訟資料。

### 目的

存取聯邦地區法院專利侵權案件。

### 關鍵資料

- 案件號和申請日期
- 主張的專利
- 當事人（原告和被告）
- 審判地
- 案件結果

### 使用案例

- 訴訟風險分析
- 識別頻繁被訴訟的專利
- 追蹤訴訟趨勢
- 分析審判地偏好
- 評估專利執法模式

## 6. 癌症登月計畫專利資料集 API

### 概述

**版本：** v1.0.1

用於癌症相關專利發現的專門資料集。

### 目的

搜尋和下載與癌症研究、治療和診斷相關的專利。

### 功能

- 經策劃的癌症相關專利
- 批量資料下載
- 按癌症類型分類
- 治療方式分類

### 使用案例

- 癌症研究先前技術
- 技術地景分析
- 識別研究趨勢
- 授權機會

## 7. OCE 專利審查狀態/事件代碼 API

### 概述

**版本：** v1

提供專利審查中使用的 USPTO 狀態和事件代碼的官方描述。

### 目的

解碼 PEDS 和其他審查資料中的交易代碼和狀態代碼。

### 提供的資料

- **狀態代碼** - 申請狀態描述
- **事件代碼** - 交易/事件描述
- **代碼定義** - 官方含義

### 整合

與 PEDS 資料一起使用以解釋交易代碼：

```python
def get_code_description(code, api_key):
    """取得 USPTO 代碼的人類可讀描述。"""
    # 從 OCE API 取得
    pass

def enrich_peds_data(peds_transactions, api_key):
    """為 PEDS 交易代碼添加描述。"""
    for trans in peds_transactions:
        trans['description'] = get_code_description(trans['code'], api_key)
    return peds_transactions
```

## API 整合模式

### 組合工作流程範例

```python
def comprehensive_patent_analysis(patent_number, api_key):
    """
    結合多個 API 的全面分析。
    """
    results = {}

    # 1. 從 PatentSearch 取得專利詳情
    results['patent_data'] = search_patent(patent_number, api_key)

    # 2. 從 PEDS 取得審查歷史
    results['prosecution'] = get_peds_data(patent_number, api_key)

    # 3. 取得轉讓歷史
    results['assignments'] = get_assignments(patent_number, api_key)

    # 4. 取得引用資料
    results['citations'] = get_citations(patent_number, api_key)

    # 5. 檢查訴訟歷史
    results['litigation'] = get_litigation(patent_number, api_key)

    # 6. 取得 PTAB 質疑
    results['ptab'] = get_ptab_proceedings(patent_number, api_key)

    return results
```

### 專利組合分析範例

```python
def analyze_company_portfolio(company_name, api_key):
    """
    使用多個 API 分析公司的專利組合。
    """
    # 1. 尋找所有轉讓的專利
    assignments = find_company_patents(company_name, api_key)
    patent_numbers = extract_patent_numbers(assignments)

    # 2. 取得每個專利的詳情
    portfolio = []
    for patent_num in patent_numbers:
        patent_data = {
            'number': patent_num,
            'details': search_patent(patent_num, api_key),
            'citations': get_citations(patent_num, api_key),
            'litigation': get_litigation(patent_num, api_key)
        }
        portfolio.append(patent_data)

    # 3. 彙總統計
    stats = {
        'total_patents': len(portfolio),
        'cited_by_count': sum(len(p['citations']) for p in portfolio),
        'litigated_count': sum(1 for p in portfolio if p['litigation']),
        'technology_areas': aggregate_tech_areas(portfolio)
    }

    return {'portfolio': portfolio, 'statistics': stats}
```

## 最佳實務

1. **API 金鑰管理** - 使用環境變數，切勿硬編碼
2. **速率限制** - 對所有 API 實作指數退避
3. **快取** - 快取 API 回應以減少重複呼叫
4. **錯誤處理** - 優雅處理 API 錯誤和缺失資料
5. **資料驗證** - 在 API 呼叫前驗證輸入格式
6. **組合 API** - 適當地組合使用 API 進行全面分析
7. **文件** - 追蹤 API 版本和變更

## API 金鑰註冊

所有 API 都需要在以下網址註冊：
**https://account.uspto.gov/api-manager/**

單一 API 金鑰適用於大多數 USPTO API。

## 資源

- **開發者入口**：https://developer.uspto.gov/
- **開放資料入口**：https://data.uspto.gov/
- **API 目錄**：https://developer.uspto.gov/api-catalog
- **Swagger 文件**：各個 API 可用
