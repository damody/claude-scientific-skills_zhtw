# Adaptyv API 參考

## 基礎 URL

```
https://kq5jp7qj7wdqklhsxmovkzn4l40obksv.lambda-url.eu-central-1.on.aws
```

## 身份驗證

所有 API 請求都需要在請求標頭中使用 bearer token 身份驗證：

```
Authorization: Bearer YOUR_API_KEY
```

取得 API 存取權限：
1. 聯繫 support@adaptyvbio.com
2. 在 alpha/beta 期間申請 API 存取權限
3. 收到您的個人存取權杖

安全儲存您的 API 金鑰：
- 使用環境變數：`ADAPTYV_API_KEY`
- 永遠不要將 API 金鑰提交到版本控制系統
- 在本地開發時使用含 `.gitignore` 的 `.env` 檔案

## 端點

### 實驗

#### 建立實驗

提交蛋白質序列進行實驗測試。

**端點：** `POST /experiments`

**請求主體：**
```json
{
  "sequences": ">protein1\nMKVLWALLGLLGAA...\n>protein2\nMATGVLWALLG...",
  "experiment_type": "binding|expression|thermostability|enzyme_activity",
  "target_id": "optional_target_identifier",
  "webhook_url": "https://your-webhook.com/callback",
  "metadata": {
    "project": "optional_project_name",
    "notes": "optional_notes"
  }
}
```

**序列格式：**
- 帶標頭的 FASTA 格式
- 支援多個序列
- 標準胺基酸代碼

**回應：**
```json
{
  "experiment_id": "exp_abc123xyz",
  "status": "submitted",
  "created_at": "2025-11-24T10:00:00Z",
  "estimated_completion": "2025-12-15T10:00:00Z"
}
```

#### 取得實驗狀態

檢查實驗的目前狀態。

**端點：** `GET /experiments/{experiment_id}`

**回應：**
```json
{
  "experiment_id": "exp_abc123xyz",
  "status": "submitted|processing|completed|failed",
  "created_at": "2025-11-24T10:00:00Z",
  "updated_at": "2025-11-25T14:30:00Z",
  "progress": {
    "stage": "sequencing|expression|assay|analysis",
    "percentage": 45
  }
}
```

**狀態值：**
- `submitted` - 實驗已收到並排隊
- `processing` - 測試正在進行中
- `completed` - 結果可供下載
- `failed` - 實驗遇到錯誤

#### 列出實驗

擷取您組織的所有實驗。

**端點：** `GET /experiments`

**查詢參數：**
- `status` - 按狀態篩選（可選）
- `limit` - 每頁結果數（預設：50）
- `offset` - 分頁偏移量（預設：0）

**回應：**
```json
{
  "experiments": [
    {
      "experiment_id": "exp_abc123xyz",
      "status": "completed",
      "experiment_type": "binding",
      "created_at": "2025-11-24T10:00:00Z"
    }
  ],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

### 結果

#### 取得實驗結果

下載已完成實驗的結果。

**端點：** `GET /experiments/{experiment_id}/results`

**回應：**
```json
{
  "experiment_id": "exp_abc123xyz",
  "results": [
    {
      "sequence_id": "protein1",
      "measurements": {
        "kd": 1.2e-9,
        "kon": 1.5e5,
        "koff": 1.8e-4
      },
      "quality_metrics": {
        "confidence": "high",
        "r_squared": 0.98
      }
    }
  ],
  "download_urls": {
    "raw_data": "https://...",
    "analysis_package": "https://...",
    "report": "https://..."
  }
}
```

### 標靶

#### 搜尋標靶目錄

搜尋 ACROBiosystems 抗原目錄。

**端點：** `GET /targets`

**查詢參數：**
- `search` - 搜尋詞（蛋白質名稱、UniProt ID 等）
- `species` - 按物種篩選
- `category` - 按類別篩選

**回應：**
```json
{
  "targets": [
    {
      "target_id": "tgt_12345",
      "name": "Human PD-L1",
      "species": "Homo sapiens",
      "uniprot_id": "Q9NZQ7",
      "availability": "in_stock|custom_order",
      "price_usd": 450
    }
  ]
}
```

#### 請求自訂標靶

請求標準目錄中沒有的抗原。

**端點：** `POST /targets/request`

**請求主體：**
```json
{
  "target_name": "Custom target name",
  "uniprot_id": "optional_uniprot_id",
  "species": "species_name",
  "notes": "Additional requirements"
}
```

### 組織

#### 取得點數餘額

檢查您組織的點數餘額和使用情況。

**端點：** `GET /organization/credits`

**回應：**
```json
{
  "balance": 10000,
  "currency": "USD",
  "usage_this_month": 2500,
  "experiments_remaining": 22
}
```

## Webhooks

配置 webhook URL 以在實驗完成時接收通知。

**Webhook 負載：**
```json
{
  "event": "experiment.completed",
  "experiment_id": "exp_abc123xyz",
  "status": "completed",
  "timestamp": "2025-12-15T10:00:00Z",
  "results_url": "/experiments/exp_abc123xyz/results"
}
```

**Webhook 事件：**
- `experiment.submitted` - 實驗已收到
- `experiment.started` - 處理已開始
- `experiment.completed` - 結果可用
- `experiment.failed` - 發生錯誤

**安全性：**
- 驗證 webhook 簽名（詳情在入門時提供）
- 僅使用 HTTPS 端點
- 回應 200 OK 以確認收到

## 錯誤處理

**錯誤回應格式：**
```json
{
  "error": {
    "code": "invalid_sequence",
    "message": "Sequence contains invalid amino acid codes",
    "details": {
      "sequence_id": "protein1",
      "position": 45,
      "character": "X"
    }
  }
}
```

**常見錯誤代碼：**
- `authentication_failed` - API 金鑰無效或缺失
- `invalid_sequence` - FASTA 格式錯誤或無效的胺基酸
- `insufficient_credits` - 點數不足以進行實驗
- `target_not_found` - 指定的標靶 ID 不存在
- `rate_limit_exceeded` - 請求過多
- `experiment_not_found` - 無效的實驗 ID
- `internal_error` - 伺服器端錯誤

## 速率限制

- 每個 API 金鑰每分鐘 100 個請求
- 每個組織每天 1000 個實驗
- 建議使用批次提交進行大規模測試

當被速率限制時，回應包含：
```
HTTP 429 Too Many Requests
Retry-After: 60
```

## 最佳實務

1. **使用 webhooks** 處理長時間執行的實驗，而非輪詢
2. **批次處理序列** 當提交多個變體時
3. **快取結果** 以避免冗餘的 API 呼叫
4. **實作重試邏輯** 並使用指數退避
5. **監控點數** 以避免實驗失敗
6. **在提交前驗證序列** 在本地進行
7. **使用描述性 metadata** 以便更好地追蹤實驗

## API 版本控制

API 目前處於 alpha/beta 階段。可能會發生破壞性變更，但會：
- 透過電子郵件通知註冊使用者
- 記錄在變更日誌中
- 提供遷移指南支援

目前版本反映在回應標頭中：
```
X-API-Version: alpha-2025-11
```

## 支援

API 問題或疑問：
- 電子郵件：support@adaptyvbio.com
- 文件更新：https://docs.adaptyvbio.com
- 回報錯誤時請附上實驗 ID 和請求詳情
