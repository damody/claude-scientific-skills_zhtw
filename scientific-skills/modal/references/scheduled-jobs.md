# 排程任務和 Cron

## 基本排程

排程函數在固定間隔或特定時間自動執行。

### 簡單的每日排程

```python
import modal

app = modal.App()

@app.function(schedule=modal.Period(days=1))
def daily_task():
    print("Running daily task")
    # 處理資料、發送報告等
```

部署以啟動：
```bash
modal deploy script.py
```

函數從部署時間起每 24 小時執行一次。

## 排程類型

### Period 排程

從部署時間起以固定間隔執行：

```python
# 每 5 小時
@app.function(schedule=modal.Period(hours=5))
def every_5_hours():
    ...

# 每 30 分鐘
@app.function(schedule=modal.Period(minutes=30))
def every_30_minutes():
    ...

# 每天
@app.function(schedule=modal.Period(days=1))
def daily():
    ...
```

**注意**：重新部署會重設週期計時器。

### Cron 排程

使用 cron 語法在特定時間執行：

```python
# 每週一上午 8 點 UTC
@app.function(schedule=modal.Cron("0 8 * * 1"))
def weekly_report():
    ...

# 每天紐約時間早上 6 點
@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def morning_report():
    ...

# 每小時整點
@app.function(schedule=modal.Cron("0 * * * *"))
def hourly():
    ...

# 每 15 分鐘
@app.function(schedule=modal.Cron("*/15 * * * *"))
def quarter_hourly():
    ...
```

**Cron 語法**：`分鐘 小時 日 月 星期幾`
- 分鐘：0-59
- 小時：0-23
- 日：1-31
- 月：1-12
- 星期幾：0-6（0 = 星期日）

### 時區支援

為 cron 排程指定時區：

```python
@app.function(schedule=modal.Cron("0 9 * * *", timezone="Europe/London"))
def uk_morning_task():
    ...

@app.function(schedule=modal.Cron("0 17 * * 5", timezone="Asia/Tokyo"))
def friday_evening_jp():
    ...
```

## 部署

### 部署排程函數

```bash
modal deploy script.py
```

排程函數持續存在直到明確停止。

### 程式化部署

```python
if __name__ == "__main__":
    app.deploy()
```

## 監控

### 檢視執行日誌

在 https://modal.com/apps 查看：
- 過去的執行日誌
- 執行歷史
- 失敗通知

### 手動執行

透過儀表板的「Run now」按鈕立即觸發排程函數。

## 排程管理

### 暫停排程

排程無法暫停。要停止：
1. 移除 `schedule` 參數
2. 重新部署應用程式

### 更新排程

變更排程參數並重新部署：

```python
# 從每日更新到每週
@app.function(schedule=modal.Period(days=7))
def task():
    ...
```

```bash
modal deploy script.py
```

## 常見模式

### 資料管線

```python
@app.function(
    schedule=modal.Cron("0 2 * * *"),  # 每日凌晨 2 點
    timeout=3600,                       # 1 小時超時
)
def etl_pipeline():
    # 從來源擷取資料
    data = extract_data()

    # 轉換資料
    transformed = transform_data(data)

    # 載入到倉儲
    load_to_warehouse(transformed)
```

### 模型重新訓練

```python
volume = modal.Volume.from_name("models")

@app.function(
    schedule=modal.Cron("0 0 * * 0"),  # 每週日午夜
    gpu="A100",
    timeout=7200,                       # 2 小時
    volumes={"/models": volume}
)
def retrain_model():
    # 載入最新資料
    data = load_training_data()

    # 訓練模型
    model = train(data)

    # 儲存新模型
    save_model(model, "/models/latest.pt")
    volume.commit()
```

### 報告生成

```python
@app.function(
    schedule=modal.Cron("0 9 * * 1"),  # 週一上午 9 點
    secrets=[modal.Secret.from_name("email-creds")]
)
def weekly_report():
    # 生成報告
    report = generate_analytics_report()

    # 發送電子郵件
    send_email(
        to="team@company.com",
        subject="Weekly Analytics Report",
        body=report
    )
```

### 資料清理

```python
@app.function(schedule=modal.Period(hours=6))
def cleanup_old_data():
    # 移除超過 30 天的資料
    cutoff = datetime.now() - timedelta(days=30)
    delete_old_records(cutoff)
```

## 使用 Secrets 和 Volumes 配置

排程函數支援所有函數參數：

```python
vol = modal.Volume.from_name("data")
secret = modal.Secret.from_name("api-keys")

@app.function(
    schedule=modal.Cron("0 */6 * * *"),  # 每 6 小時
    secrets=[secret],
    volumes={"/data": vol},
    cpu=4.0,
    memory=16384,
)
def sync_data():
    import os

    api_key = os.environ["API_KEY"]

    # 從外部 API 取得
    data = fetch_external_data(api_key)

    # 儲存到 volume
    with open("/data/latest.json", "w") as f:
        json.dump(data, f)

    vol.commit()
```

## 動態排程

以程式化方式更新排程：

```python
@app.function()
def main_task():
    ...

@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def enable_high_traffic_mode():
    main_task.update_autoscaler(min_containers=5)

@app.function(schedule=modal.Cron("0 22 * * *", timezone="America/New_York"))
def disable_high_traffic_mode():
    main_task.update_autoscaler(min_containers=0)
```

## 錯誤處理

失敗的排程函數會：
- 在儀表板顯示失敗
- 發送通知（可配置）
- 在下次排程執行時重試

```python
@app.function(
    schedule=modal.Cron("0 * * * *"),
    retries=3,  # 重試失敗的執行
    timeout=1800
)
def robust_task():
    try:
        perform_task()
    except Exception as e:
        # 記錄錯誤
        print(f"Task failed: {e}")
        # 可選發送警報
        send_alert(f"Scheduled task failed: {e}")
        raise
```

## 最佳實踐

1. **設定超時**：始終為排程函數指定超時
2. **使用適當的排程**：Period 用於相對時間，Cron 用於絕對時間
3. **監控失敗**：定期查看儀表板的失敗執行
4. **冪等操作**：設計任務以安全處理重複執行
5. **資源限制**：為排程工作負載設定適當的 CPU/記憶體
6. **時區意識**：為 cron 排程指定時區
