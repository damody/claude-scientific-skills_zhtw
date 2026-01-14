# 連線與會話管理

此參考涵蓋使用 BlitzGateway 建立和管理與 OMERO 伺服器的連線。

## 基本連線

### 標準連線模式

```python
from omero.gateway import BlitzGateway

# 建立連線
conn = BlitzGateway(username, password, host=host, port=4064)

# 連線到伺服器
if conn.connect():
    print("Connected successfully")
    # 執行操作
    conn.close()
else:
    print("Failed to connect")
```

### 連線參數

- **username** (str)：OMERO 使用者帳號名稱
- **password** (str)：使用者密碼
- **host** (str)：OMERO 伺服器主機名稱或 IP 位址
- **port** (int)：伺服器連接埠（預設：4064）
- **secure** (bool)：強制加密連線（預設：False）

### 安全連線

確保所有資料傳輸都加密：

```python
conn = BlitzGateway(username, password, host=host, port=4064, secure=True)
conn.connect()
```

## 上下文管理器模式（推薦）

使用上下文管理器進行自動連線管理和清理：

```python
from omero.gateway import BlitzGateway

with BlitzGateway(username, password, host=host, port=4064) as conn:
    # 連線自動建立
    for project in conn.getObjects('Project'):
        print(project.getName())
    # 退出時自動關閉連線
```

**優點：**
- 自動呼叫 `connect()`
- 退出時自動呼叫 `close()`
- 異常安全的資源清理
- 更簡潔的程式碼

## 會話管理

### 從現有用戶端建立連線

從現有的 `omero.client` 會話建立 BlitzGateway：

```python
import omero.clients
from omero.gateway import BlitzGateway

# 建立用戶端和會話
client = omero.client(host, port)
session = client.createSession(username, password)

# 從現有用戶端建立 BlitzGateway
conn = BlitzGateway(client_obj=client)

# 使用連線
# ...

# 完成後關閉
conn.close()
```

### 擷取會話資訊

```python
# 取得當前使用者資訊
user = conn.getUser()
print(f"User ID: {user.getId()}")
print(f"Username: {user.getName()}")
print(f"Full Name: {user.getFullName()}")
print(f"Is Admin: {conn.isAdmin()}")

# 取得當前群組
group = conn.getGroupFromContext()
print(f"Current Group: {group.getName()}")
print(f"Group ID: {group.getId()}")
```

### 檢查管理員權限

```python
if conn.isAdmin():
    print("User has admin privileges")

if conn.isFullAdmin():
    print("User is full administrator")
else:
    # 檢查特定管理員權限
    privileges = conn.getCurrentAdminPrivileges()
    print(f"Admin privileges: {privileges}")
```

## 群組上下文管理

OMERO 使用群組來管理資料存取權限。使用者可以屬於多個群組。

### 取得當前群組上下文

```python
# 取得當前群組上下文
group = conn.getGroupFromContext()
print(f"Current group: {group.getName()}")
print(f"Group ID: {group.getId()}")
```

### 跨所有群組查詢

使用群組 ID `-1` 跨所有可存取的群組查詢：

```python
# 設定上下文以查詢所有群組
conn.SERVICE_OPTS.setOmeroGroup('-1')

# 現在查詢跨越所有可存取的群組
image = conn.getObject("Image", image_id)
projects = conn.listProjects()
```

### 切換到特定群組

切換上下文以在特定群組中工作：

```python
# 從物件取得群組 ID
image = conn.getObject("Image", image_id)
group_id = image.getDetails().getGroup().getId()

# 切換到該群組的上下文
conn.SERVICE_OPTS.setOmeroGroup(group_id)

# 後續操作使用此群組上下文
projects = conn.listProjects()
```

### 列出可用群組

```python
# 取得當前使用者的所有群組
for group in conn.getGroupsMemberOf():
    print(f"Group: {group.getName()} (ID: {group.getId()})")
```

## 進階連線功能

### 替代使用者連線（僅限管理員）

管理員可以建立以其他使用者身份操作的連線：

```python
# 以管理員身份連線
admin_conn = BlitzGateway(admin_user, admin_pass, host=host, port=4064)
admin_conn.connect()

# 取得目標使用者
target_user = admin_conn.getObject("Experimenter", user_id).getName()

# 建立該使用者的連線
user_conn = admin_conn.suConn(target_user)

# 以目標使用者身份執行操作
for project in user_conn.listProjects():
    print(project.getName())

# 關閉替代連線
user_conn.close()
admin_conn.close()
```

### 列出管理員

```python
# 取得所有管理員
for admin in conn.getAdministrators():
    print(f"ID: {admin.getId()}, Name: {admin.getFullName()}, "
          f"Username: {admin.getOmeName()}")
```

## 連線生命週期

### 關閉連線

務必關閉連線以釋放伺服器資源：

```python
try:
    conn = BlitzGateway(username, password, host=host, port=4064)
    conn.connect()

    # 執行操作

except Exception as e:
    print(f"Error: {e}")
finally:
    if conn:
        conn.close()
```

### 檢查連線狀態

```python
if conn.isConnected():
    print("Connection is active")
else:
    print("Connection is closed")
```

## 錯誤處理

### 穩健的連線模式

```python
from omero.gateway import BlitzGateway
import traceback

def connect_to_omero(username, password, host, port=4064):
    """
    建立與 OMERO 伺服器的連線，具有錯誤處理。

    傳回：
        BlitzGateway 連線物件，如果失敗則傳回 None
    """
    try:
        conn = BlitzGateway(username, password, host=host, port=port, secure=True)
        if conn.connect():
            print(f"Connected to {host}:{port} as {username}")
            return conn
        else:
            print("Failed to establish connection")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        traceback.print_exc()
        return None

# 使用方式
conn = connect_to_omero(username, password, host)
if conn:
    try:
        # 執行操作
        pass
    finally:
        conn.close()
```

## 常見連線模式

### 模式 1：簡單腳本

```python
from omero.gateway import BlitzGateway

# 連線參數
HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

# 連線
with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    print(f"Connected as {conn.getUser().getName()}")
    # 執行操作
```

### 模式 2：基於配置的連線

```python
import yaml
from omero.gateway import BlitzGateway

# 載入配置
with open('omero_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 使用配置連線
with BlitzGateway(
    config['username'],
    config['password'],
    host=config['host'],
    port=config.get('port', 4064),
    secure=config.get('secure', True)
) as conn:
    # 執行操作
    pass
```

### 模式 3：環境變數

```python
import os
from omero.gateway import BlitzGateway

# 從環境取得憑證
USERNAME = os.environ.get('OMERO_USER')
PASSWORD = os.environ.get('OMERO_PASSWORD')
HOST = os.environ.get('OMERO_HOST', 'localhost')
PORT = int(os.environ.get('OMERO_PORT', 4064))

# 連線
with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 執行操作
    pass
```

## 最佳實務

1. **使用上下文管理器**：務必優先使用上下文管理器進行自動清理
2. **安全連線**：在生產環境中使用 `secure=True`
3. **錯誤處理**：將連線程式碼包裝在 try-except 區塊中
4. **關閉連線**：完成後務必關閉連線
5. **群組上下文**：在查詢前設定適當的群組上下文
6. **憑證安全**：切勿硬編碼憑證；使用環境變數或配置檔
7. **連線池**：對於 Web 應用程式，實作連線池
8. **逾時**：考慮為長時間執行的操作實作連線逾時

## 疑難排解

### 連線被拒絕

```
Unable to contact ORB
```

**解決方案：**
- 驗證主機和連接埠是否正確
- 檢查防火牆設定
- 確保 OMERO 伺服器正在執行
- 驗證網路連通性

### 身份驗證失敗

```
Cannot connect to server
```

**解決方案：**
- 驗證使用者名稱和密碼
- 檢查使用者帳號是否啟用
- 驗證群組成員資格
- 檢查伺服器日誌以取得詳情

### 會話逾時

**解決方案：**
- 增加伺服器上的會話逾時
- 實作會話保活機制
- 逾時時重新連線
- 對長時間執行的應用程式使用連線池
