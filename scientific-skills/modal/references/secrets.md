# 密鑰和環境變數

## 建立密鑰

### 透過儀表板

在 https://modal.com/secrets 建立密鑰

可用範本：
- 資料庫憑證（Postgres、MongoDB）
- 雲端提供商（AWS、GCP、Azure）
- ML 平台（Weights & Biases、Hugging Face）
- 以及更多

### 透過 CLI

```bash
# 使用鍵值對建立密鑰
modal secret create my-secret KEY1=value1 KEY2=value2

# 使用環境變數
modal secret create db-secret PGHOST=uri PGPASSWORD="$PGPASSWORD"

# 列出密鑰
modal secret list

# 刪除密鑰
modal secret delete my-secret
```

### 程式化方式

從字典：

```python
if modal.is_local():
    local_secret = modal.Secret.from_dict({"FOO": os.environ["LOCAL_FOO"]})
else:
    local_secret = modal.Secret.from_dict({})

@app.function(secrets=[local_secret])
def some_function():
    import os
    print(os.environ["FOO"])
```

從 .env 檔案：

```python
@app.function(secrets=[modal.Secret.from_dotenv()])
def some_function():
    import os
    print(os.environ["USERNAME"])
```

## 使用密鑰

將密鑰注入函數：

```python
@app.function(secrets=[modal.Secret.from_name("my-secret")])
def some_function():
    import os
    secret_key = os.environ["MY_PASSWORD"]
    # 使用密鑰
    ...
```

### 多個密鑰

```python
@app.function(secrets=[
    modal.Secret.from_name("database-creds"),
    modal.Secret.from_name("api-keys"),
])
def other_function():
    # 兩個密鑰的所有鍵都可用
    ...
```

如果鍵衝突，後面的密鑰會覆蓋前面的。

## 環境變數

### 保留的執行時變數

**所有容器**：
- `MODAL_CLOUD_PROVIDER` - 雲端提供商（AWS/GCP/OCI）
- `MODAL_IMAGE_ID` - 映像 ID
- `MODAL_REGION` - 區域識別碼（例如：us-east-1）
- `MODAL_TASK_ID` - 容器任務 ID

**函數容器**：
- `MODAL_ENVIRONMENT` - Modal 環境名稱
- `MODAL_IS_REMOTE` - 在遠端容器中設為 '1'
- `MODAL_IDENTITY_TOKEN` - 函數身份的 OIDC 權杖

**沙箱容器**：
- `MODAL_SANDBOX_ID` - 沙箱 ID

### 設定環境變數

透過映像：

```python
image = modal.Image.debian_slim().env({"PORT": "6443"})

@app.function(image=image)
def my_function():
    import os
    port = os.environ["PORT"]
```

透過密鑰：

```python
secret = modal.Secret.from_dict({"API_KEY": "secret-value"})

@app.function(secrets=[secret])
def my_function():
    import os
    api_key = os.environ["API_KEY"]
```

## 常見密鑰模式

### AWS 憑證

```python
aws_secret = modal.Secret.from_name("my-aws-secret")

@app.function(secrets=[aws_secret])
def use_aws():
    import boto3
    s3 = boto3.client('s3')
    # AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY 自動使用
```

### Hugging Face 權杖

```python
hf_secret = modal.Secret.from_name("huggingface")

@app.function(secrets=[hf_secret])
def download_model():
    from transformers import AutoModel
    # HF_TOKEN 自動用於驗證
    model = AutoModel.from_pretrained("private-model")
```

### 資料庫憑證

```python
db_secret = modal.Secret.from_name("postgres-creds")

@app.function(secrets=[db_secret])
def query_db():
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )
```

## 最佳實踐

1. **永遠不要硬編碼密鑰** - 始終使用 Modal Secrets
2. **使用特定密鑰** - 為不同目的建立不同的密鑰
3. **定期輪換密鑰** - 定期更新密鑰
4. **最小範圍** - 只將密鑰附加到需要它們的函數
5. **環境特定** - 為 dev/staging/prod 使用不同的密鑰

## 安全說明

- 密鑰在靜止時加密
- 只有明確請求的函數才能存取
- 不會記錄或在儀表板中顯示
- 可以限定到特定環境
