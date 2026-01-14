# 進階功能

此參考涵蓋進階 OMERO 操作，包括權限、刪除、檔案集和管理任務。

## 刪除物件

### 等待刪除完成

```python
# 刪除物件並等待完成
project_ids = [1, 2, 3]
conn.deleteObjects("Project", project_ids, wait=True)
print("Deletion complete")

# 不等待即刪除（非同步）
conn.deleteObjects("Dataset", [dataset_id], wait=False)
```

### 使用回呼監控刪除

```python
from omero.callbacks import CmdCallbackI

# 開始刪除操作
handle = conn.deleteObjects("Project", [project_id])

# 建立回呼以監控進度
cb = CmdCallbackI(conn.c, handle)
print("Deleting, please wait...")

# 輪詢完成狀態
while not cb.block(500):  # 每 500ms 檢查一次
    print(".", end="", flush=True)

print("\nDeletion finished")

# 檢查錯誤
response = cb.getResponse()
if isinstance(response, omero.cmd.ERR):
    print("Error occurred:")
    print(response)
else:
    print("Deletion successful")

# 清理
cb.close(True)  # 同時關閉 handle
```

### 刪除不同物件類型

```python
# 刪除影像
image_ids = [101, 102, 103]
conn.deleteObjects("Image", image_ids, wait=True)

# 刪除資料集
dataset_ids = [10, 11]
conn.deleteObjects("Dataset", dataset_ids, wait=True)

# 刪除 ROI
roi_ids = [201, 202]
conn.deleteObjects("Roi", roi_ids, wait=True)

# 刪除註解
annotation_ids = [301, 302]
conn.deleteObjects("Annotation", annotation_ids, wait=True)
```

### 級聯刪除

```python
# 刪除專案將級聯到包含的資料集
# 此行為取決於伺服器配置
project_id = 123
conn.deleteObjects("Project", [project_id], wait=True)

# 資料集和影像可能被刪除或成為孤立
# 取決於刪除規範
```

## 檔案集

檔案集代表原始匯入檔案的集合。它們是在 OMERO 5.0 中引入的。

### 檢查影像是否有檔案集

```python
image = conn.getObject("Image", image_id)

fileset = image.getFileset()
if fileset:
    print(f"Image is part of fileset {fileset.getId()}")
else:
    print("Image has no fileset (pre-OMERO 5.0)")
```

### 存取檔案集資訊

```python
image = conn.getObject("Image", image_id)
fileset = image.getFileset()

if fileset:
    fs_id = fileset.getId()
    print(f"Fileset ID: {fs_id}")

    # 列出此檔案集中的所有影像
    print("Images in fileset:")
    for fs_image in fileset.copyImages():
        print(f"  {fs_image.getId()}: {fs_image.getName()}")

    # 列出原始匯入檔案
    print("\nOriginal files:")
    for orig_file in fileset.listFiles():
        print(f"  {orig_file.getPath()}/{orig_file.getName()}")
        print(f"    Size: {orig_file.getSize()} bytes")
```

### 直接取得檔案集

```python
# 取得檔案集物件
fileset = conn.getObject("Fileset", fileset_id)

if fileset:
    # 存取影像
    for image in fileset.copyImages():
        print(f"Image: {image.getName()}")

    # 存取檔案
    for orig_file in fileset.listFiles():
        print(f"File: {orig_file.getName()}")
```

### 下載原始檔案

```python
import os

fileset = image.getFileset()

if fileset:
    download_dir = "./original_files"
    os.makedirs(download_dir, exist_ok=True)

    for orig_file in fileset.listFiles():
        file_name = orig_file.getName()
        file_path = os.path.join(download_dir, file_name)

        print(f"Downloading: {file_name}")

        # 以 RawFileStore 取得檔案
        raw_file_store = conn.createRawFileStore()
        raw_file_store.setFileId(orig_file.getId())

        # 分塊下載
        with open(file_path, 'wb') as f:
            offset = 0
            chunk_size = 1024 * 1024  # 1MB 塊
            size = orig_file.getSize()

            while offset < size:
                chunk = raw_file_store.read(offset, chunk_size)
                f.write(chunk)
                offset += len(chunk)

        raw_file_store.close()
        print(f"Saved to: {file_path}")
```

## 群組權限

OMERO 使用基於群組的權限來控制資料存取。

### 權限等級

- **PRIVATE** (`rw----`)：只有所有者可以讀取/寫入
- **READ-ONLY** (`rwr---`)：群組成員可以讀取，只有所有者可以寫入
- **READ-ANNOTATE** (`rwra--`)：群組成員可以讀取和註解
- **READ-WRITE** (`rwrw--`)：群組成員可以讀取和寫入

### 檢查當前群組權限

```python
# 取得當前群組
group = conn.getGroupFromContext()

# 取得權限
permissions = group.getDetails().getPermissions()
perm_string = str(permissions)

# 對應到可讀名稱
permission_names = {
    'rw----': 'PRIVATE',
    'rwr---': 'READ-ONLY',
    'rwra--': 'READ-ANNOTATE',
    'rwrw--': 'READ-WRITE'
}

perm_name = permission_names.get(perm_string, 'UNKNOWN')
print(f"Group: {group.getName()}")
print(f"Permissions: {perm_name} ({perm_string})")
```

### 列出使用者的群組

```python
# 取得當前使用者的所有群組
print("User's groups:")
for group in conn.getGroupsMemberOf():
    print(f"  {group.getName()} (ID: {group.getId()})")

    # 取得群組權限
    perms = group.getDetails().getPermissions()
    print(f"    Permissions: {perms}")
```

### 取得群組成員

```python
# 取得群組
group = conn.getObject("ExperimenterGroup", group_id)

# 列出成員
print(f"Members of {group.getName()}:")
for member in group.getMembers():
    print(f"  {member.getFullName()} ({member.getOmeName()})")
```

## 跨群組查詢

### 查詢所有群組

```python
# 設定上下文以查詢所有可存取的群組
conn.SERVICE_OPTS.setOmeroGroup('-1')

# 現在查詢跨越所有群組
image = conn.getObject("Image", image_id)
if image:
    group = image.getDetails().getGroup()
    print(f"Image found in group: {group.getName()}")

# 列出所有群組中的專案
for project in conn.getObjects("Project"):
    group = project.getDetails().getGroup()
    print(f"Project: {project.getName()} (Group: {group.getName()})")
```

### 切換到特定群組

```python
# 取得影像的群組
image = conn.getObject("Image", image_id)
group_id = image.getDetails().getGroup().getId()

# 切換到該群組的上下文
conn.SERVICE_OPTS.setOmeroGroup(group_id)

# 後續操作使用此群組
projects = conn.listProjects()  # 只來自此群組
```

### 重設為預設群組

```python
# 取得預設群組
default_group_id = conn.getEventContext().groupId

# 切換回預設
conn.SERVICE_OPTS.setOmeroGroup(default_group_id)
```

## 管理操作

### 檢查管理員狀態

```python
# 檢查當前使用者是否為管理員
if conn.isAdmin():
    print("User has admin privileges")

# 檢查是否為完整管理員
if conn.isFullAdmin():
    print("User is full administrator")
else:
    # 檢查特定權限
    privileges = conn.getCurrentAdminPrivileges()
    print(f"Admin privileges: {privileges}")
```

### 列出管理員

```python
# 取得所有管理員
print("Administrators:")
for admin in conn.getAdministrators():
    print(f"  ID: {admin.getId()}")
    print(f"  Username: {admin.getOmeName()}")
    print(f"  Full Name: {admin.getFullName()}")
```

### 設定物件所有者（僅限管理員）

```python
import omero.model

# 建立具有特定所有者的註解（需要管理員權限）
tag_ann = omero.gateway.TagAnnotationWrapper(conn)
tag_ann.setValue("Admin-created tag")

# 設定所有者
user_id = 5
tag_ann._obj.details.owner = omero.model.ExperimenterI(user_id, False)
tag_ann.save()

print(f"Created annotation owned by user {user_id}")
```

### 替代使用者連線（僅限管理員）

```python
# 以管理員身份連線
admin_conn = BlitzGateway(admin_user, admin_pass, host=host, port=4064)
admin_conn.connect()

# 取得目標使用者
target_user_id = 10
user = admin_conn.getObject("Experimenter", target_user_id)
username = user.getOmeName()

# 建立該使用者的連線
user_conn = admin_conn.suConn(username)

print(f"Connected as {username}")

# 以該使用者身份執行操作
for project in user_conn.listProjects():
    print(f"  {project.getName()}")

# 關閉連線
user_conn.close()
admin_conn.close()
```

### 列出所有使用者

```python
# 取得所有使用者（管理員操作）
print("All users:")
for user in conn.getObjects("Experimenter"):
    print(f"  ID: {user.getId()}")
    print(f"  Username: {user.getOmeName()}")
    print(f"  Full Name: {user.getFullName()}")
    print(f"  Email: {user.getEmail()}")
    print()
```

## 服務存取

OMERO 提供各種服務用於特定操作。

### 更新服務

```python
# 取得更新服務
updateService = conn.getUpdateService()

# 儲存並傳回物件
roi = omero.model.RoiI()
roi.setImage(image._obj)
saved_roi = updateService.saveAndReturnObject(roi)

# 儲存多個物件
objects = [obj1, obj2, obj3]
saved_objects = updateService.saveAndReturnArray(objects)
```

### ROI 服務

```python
# 取得 ROI 服務
roi_service = conn.getRoiService()

# 尋找影像的 ROI
result = roi_service.findByImage(image_id, None)

# 取得形狀統計
shape_ids = [shape.id.val for roi in result.rois
             for shape in roi.copyShapes()]
stats = roi_service.getShapeStatsRestricted(shape_ids, 0, 0, [0])
```

### 中繼資料服務

```python
# 取得中繼資料服務
metadataService = conn.getMetadataService()

# 按類型和命名空間載入註解
ns_to_include = ["mylab.analysis"]
ns_to_exclude = []

annotations = metadataService.loadSpecifiedAnnotations(
    'omero.model.FileAnnotation',
    ns_to_include,
    ns_to_exclude,
    None
)

for ann in annotations:
    print(f"Annotation: {ann.getId().getValue()}")
```

### 查詢服務

```python
# 取得查詢服務
queryService = conn.getQueryService()

# 建立查詢（更複雜的查詢）
params = omero.sys.ParametersI()
params.addLong("image_id", image_id)

query = "select i from Image i where i.id = :image_id"
image = queryService.findByQuery(query, params)
```

### 縮圖服務

```python
# 取得縮圖服務
thumbnailService = conn.createThumbnailStore()

# 設定當前影像
thumbnailService.setPixelsId(image.getPrimaryPixels().getId())

# 取得縮圖
thumbnail = thumbnailService.getThumbnail(96, 96)

# 關閉服務
thumbnailService.close()
```

### 原始檔案儲存

```python
# 取得原始檔案儲存
rawFileStore = conn.createRawFileStore()

# 設定檔案 ID
rawFileStore.setFileId(orig_file_id)

# 讀取檔案
data = rawFileStore.read(0, rawFileStore.size())

# 關閉
rawFileStore.close()
```

## 物件所有權和詳情

### 取得物件詳情

```python
image = conn.getObject("Image", image_id)

# 取得詳情
details = image.getDetails()

# 所有者資訊
owner = details.getOwner()
print(f"Owner ID: {owner.getId()}")
print(f"Username: {owner.getOmeName()}")
print(f"Full Name: {owner.getFullName()}")

# 群組資訊
group = details.getGroup()
print(f"Group: {group.getName()} (ID: {group.getId()})")

# 建立資訊
creation_event = details.getCreationEvent()
print(f"Created: {creation_event.getTime()}")

# 更新資訊
update_event = details.getUpdateEvent()
print(f"Updated: {update_event.getTime()}")
```

### 取得權限

```python
# 取得物件權限
details = image.getDetails()
permissions = details.getPermissions()

# 檢查特定權限
can_edit = permissions.canEdit()
can_annotate = permissions.canAnnotate()
can_link = permissions.canLink()
can_delete = permissions.canDelete()

print(f"Can edit: {can_edit}")
print(f"Can annotate: {can_annotate}")
print(f"Can link: {can_link}")
print(f"Can delete: {can_delete}")
```

## 事件上下文

### 取得當前事件上下文

```python
# 取得事件上下文（當前會話資訊）
ctx = conn.getEventContext()

print(f"User ID: {ctx.userId}")
print(f"Username: {ctx.userName}")
print(f"Group ID: {ctx.groupId}")
print(f"Group Name: {ctx.groupName}")
print(f"Session ID: {ctx.sessionId}")
print(f"Is Admin: {ctx.isAdmin}")
```

## 完整管理員範例

```python
from omero.gateway import BlitzGateway

# 以管理員身份連線
ADMIN_USER = 'root'
ADMIN_PASS = 'password'
HOST = 'omero.example.com'
PORT = 4064

with BlitzGateway(ADMIN_USER, ADMIN_PASS, host=HOST, port=PORT) as admin_conn:
    print("=== Administrator Operations ===\n")

    # 列出所有使用者
    print("All Users:")
    for user in admin_conn.getObjects("Experimenter"):
        print(f"  {user.getOmeName()}: {user.getFullName()}")

    # 列出所有群組
    print("\nAll Groups:")
    for group in admin_conn.getObjects("ExperimenterGroup"):
        perms = group.getDetails().getPermissions()
        print(f"  {group.getName()}: {perms}")

        # 列出成員
        for member in group.getMembers():
            print(f"    - {member.getOmeName()}")

    # 跨所有群組查詢
    print("\nAll Projects (all groups):")
    admin_conn.SERVICE_OPTS.setOmeroGroup('-1')

    for project in admin_conn.getObjects("Project"):
        owner = project.getDetails().getOwner()
        group = project.getDetails().getGroup()
        print(f"  {project.getName()}")
        print(f"    Owner: {owner.getOmeName()}")
        print(f"    Group: {group.getName()}")

    # 以另一使用者身份連線
    target_user_id = 5
    user = admin_conn.getObject("Experimenter", target_user_id)

    if user:
        print(f"\n=== Operating as {user.getOmeName()} ===\n")

        user_conn = admin_conn.suConn(user.getOmeName())

        # 列出該使用者的專案
        for project in user_conn.listProjects():
            print(f"  {project.getName()}")

        user_conn.close()
```

## 最佳實務

1. **權限**：在操作前務必檢查權限
2. **群組上下文**：為查詢設定適當的群組上下文
3. **管理員操作**：謹慎且節制地使用管理員權限
4. **刪除確認**：在刪除物件前務必確認
5. **回呼監控**：使用回呼監控長時間的刪除操作
6. **檔案集意識**：處理影像時檢查檔案集
7. **服務清理**：完成後關閉服務（thumbnailStore、rawFileStore）
8. **跨群組查詢**：使用 `-1` 群組 ID 進行跨群組存取
9. **錯誤處理**：務必處理權限和存取錯誤
10. **文件記錄**：清楚記錄管理操作

## 疑難排解

### 權限被拒絕

```python
try:
    conn.deleteObjects("Project", [project_id], wait=True)
except Exception as e:
    if "SecurityViolation" in str(e):
        print("Permission denied: You don't own this object")
    else:
        raise
```

### 物件未找到

```python
# 存取前檢查物件是否存在
obj = conn.getObject("Image", image_id)
if obj is None:
    print(f"Image {image_id} not found or not accessible")
else:
    # 處理物件
    pass
```

### 群組上下文問題

```python
# 如果物件未找到，嘗試跨群組查詢
conn.SERVICE_OPTS.setOmeroGroup('-1')
obj = conn.getObject("Image", image_id)

if obj:
    # 切換到物件的群組以進行後續操作
    group_id = obj.getDetails().getGroup().getId()
    conn.SERVICE_OPTS.setOmeroGroup(group_id)
```
