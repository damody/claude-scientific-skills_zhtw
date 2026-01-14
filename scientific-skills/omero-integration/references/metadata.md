# 中繼資料與註解

此參考涵蓋在 OMERO 中建立和管理註解，包括標籤、鍵值對、檔案附件和評論。

## 註解類型

OMERO 支援多種註解類型：

- **TagAnnotation**：用於分類的文字標籤
- **MapAnnotation**：結構化中繼資料的鍵值對
- **FileAnnotation**：檔案附件（PDF、CSV、分析結果等）
- **CommentAnnotation**：自由格式文字評論
- **LongAnnotation**：整數值
- **DoubleAnnotation**：浮點數值
- **BooleanAnnotation**：布林值
- **TimestampAnnotation**：日期/時間戳記
- **TermAnnotation**：本體論術語

## 標籤註解

### 建立並連結標籤

```python
import omero.gateway

# 建立新標籤
tag_ann = omero.gateway.TagAnnotationWrapper(conn)
tag_ann.setValue("Experiment 2024")
tag_ann.setDescription("Optional description of this tag")
tag_ann.save()

# 將標籤連結到物件
project = conn.getObject("Project", project_id)
project.linkAnnotation(tag_ann)
```

### 建立具有命名空間的標籤

```python
# 建立具有自訂命名空間的標籤
tag_ann = omero.gateway.TagAnnotationWrapper(conn)
tag_ann.setValue("Quality Control")
tag_ann.setNs("mylab.qc.tags")
tag_ann.save()

# 連結到影像
image = conn.getObject("Image", image_id)
image.linkAnnotation(tag_ann)
```

### 重用現有標籤

```python
# 尋找現有標籤
tag_id = 123
tag_ann = conn.getObject("TagAnnotation", tag_id)

# 連結到多張影像
for image in conn.getObjects("Image", [img1, img2, img3]):
    image.linkAnnotation(tag_ann)
```

### 建立標籤集（具有子標籤的標籤）

```python
# 建立標籤集（父標籤）
tag_set = omero.gateway.TagAnnotationWrapper(conn)
tag_set.setValue("Cell Types")
tag_set.save()

# 建立子標籤
tags = ["HeLa", "U2OS", "HEK293"]
for tag_value in tags:
    tag = omero.gateway.TagAnnotationWrapper(conn)
    tag.setValue(tag_value)
    tag.save()

    # 將子標籤連結到父標籤
    tag_set.linkAnnotation(tag)
```

## 對應註解（鍵值對）

### 建立對應註解

```python
import omero.gateway
import omero.constants.metadata

# 準備鍵值資料
key_value_data = [
    ["Drug Name", "Monastrol"],
    ["Concentration", "5 mg/ml"],
    ["Treatment Time", "24 hours"],
    ["Temperature", "37C"]
]

# 建立對應註解
map_ann = omero.gateway.MapAnnotationWrapper(conn)

# 使用標準用戶端命名空間
namespace = omero.constants.metadata.NSCLIENTMAPANNOTATION
map_ann.setNs(namespace)

# 設定資料
map_ann.setValue(key_value_data)
map_ann.save()

# 連結到資料集
dataset = conn.getObject("Dataset", dataset_id)
dataset.linkAnnotation(map_ann)
```

### 對應註解的自訂命名空間

```python
# 使用自訂命名空間用於組織特定的中繼資料
key_value_data = [
    ["Microscope", "Zeiss LSM 880"],
    ["Objective", "63x Oil"],
    ["Laser Power", "10%"]
]

map_ann = omero.gateway.MapAnnotationWrapper(conn)
map_ann.setNs("mylab.microscopy.settings")
map_ann.setValue(key_value_data)
map_ann.save()

image = conn.getObject("Image", image_id)
image.linkAnnotation(map_ann)
```

### 讀取對應註解

```python
# 取得對應註解
image = conn.getObject("Image", image_id)

for ann in image.listAnnotations():
    if isinstance(ann, omero.gateway.MapAnnotationWrapper):
        print(f"Map Annotation (ID: {ann.getId()}):")
        print(f"Namespace: {ann.getNs()}")

        # 取得鍵值對
        for key, value in ann.getValue():
            print(f"  {key}: {value}")
```

## 檔案註解

### 上傳並附加檔案

```python
import os

# 準備檔案
file_path = "analysis_results.csv"

# 建立檔案註解
namespace = "mylab.analysis.results"
file_ann = conn.createFileAnnfromLocalFile(
    file_path,
    mimetype="text/csv",
    ns=namespace,
    desc="Cell segmentation results"
)

# 連結到資料集
dataset = conn.getObject("Dataset", dataset_id)
dataset.linkAnnotation(file_ann)
```

### 支援的 MIME 類型

常見 MIME 類型：
- 文字：`"text/plain"`、`"text/csv"`、`"text/tab-separated-values"`
- 文件：`"application/pdf"`、`"application/vnd.ms-excel"`
- 影像：`"image/png"`、`"image/jpeg"`
- 資料：`"application/json"`、`"application/xml"`
- 壓縮檔：`"application/zip"`、`"application/gzip"`

### 上傳多個檔案

```python
files = ["figure1.pdf", "figure2.pdf", "table1.csv"]
namespace = "publication.supplementary"

dataset = conn.getObject("Dataset", dataset_id)

for file_path in files:
    file_ann = conn.createFileAnnfromLocalFile(
        file_path,
        mimetype="application/octet-stream",
        ns=namespace,
        desc=f"Supplementary file: {os.path.basename(file_path)}"
    )
    dataset.linkAnnotation(file_ann)
```

### 下載檔案註解

```python
import os

# 取得具有檔案註解的物件
image = conn.getObject("Image", image_id)

# 下載目錄
download_path = "./downloads"
os.makedirs(download_path, exist_ok=True)

# 按命名空間篩選
namespace = "mylab.analysis.results"

for ann in image.listAnnotations(ns=namespace):
    if isinstance(ann, omero.gateway.FileAnnotationWrapper):
        file_name = ann.getFile().getName()
        file_path = os.path.join(download_path, file_name)

        print(f"Downloading: {file_name}")

        # 分塊下載檔案
        with open(file_path, 'wb') as f:
            for chunk in ann.getFileInChunks():
                f.write(chunk)

        print(f"Saved to: {file_path}")
```

### 取得檔案註解中繼資料

```python
for ann in dataset.listAnnotations():
    if isinstance(ann, omero.gateway.FileAnnotationWrapper):
        orig_file = ann.getFile()

        print(f"File Annotation ID: {ann.getId()}")
        print(f"  File Name: {orig_file.getName()}")
        print(f"  File Size: {orig_file.getSize()} bytes")
        print(f"  MIME Type: {orig_file.getMimetype()}")
        print(f"  Namespace: {ann.getNs()}")
        print(f"  Description: {ann.getDescription()}")
```

## 評論註解

### 新增評論

```python
# 建立評論
comment = omero.gateway.CommentAnnotationWrapper(conn)
comment.setValue("This image shows excellent staining quality")
comment.save()

# 連結到影像
image = conn.getObject("Image", image_id)
image.linkAnnotation(comment)
```

### 新增具有命名空間的評論

```python
comment = omero.gateway.CommentAnnotationWrapper(conn)
comment.setValue("Approved for publication")
comment.setNs("mylab.publication.status")
comment.save()

dataset = conn.getObject("Dataset", dataset_id)
dataset.linkAnnotation(comment)
```

## 數值註解

### Long 註解（整數）

```python
# 建立 long 註解
long_ann = omero.gateway.LongAnnotationWrapper(conn)
long_ann.setValue(42)
long_ann.setNs("mylab.cell.count")
long_ann.save()

image = conn.getObject("Image", image_id)
image.linkAnnotation(long_ann)
```

### Double 註解（浮點數）

```python
# 建立 double 註解
double_ann = omero.gateway.DoubleAnnotationWrapper(conn)
double_ann.setValue(3.14159)
double_ann.setNs("mylab.fluorescence.intensity")
double_ann.save()

image = conn.getObject("Image", image_id)
image.linkAnnotation(double_ann)
```

## 列出註解

### 列出物件上的所有註解

```python
import omero.model

# 取得物件
project = conn.getObject("Project", project_id)

# 列出所有註解
for ann in project.listAnnotations():
    print(f"Annotation ID: {ann.getId()}")
    print(f"  Type: {ann.OMERO_TYPE}")
    print(f"  Added by: {ann.link.getDetails().getOwner().getOmeName()}")

    # 類型特定處理
    if ann.OMERO_TYPE == omero.model.TagAnnotationI:
        print(f"  Tag value: {ann.getTextValue()}")

    elif isinstance(ann, omero.gateway.MapAnnotationWrapper):
        print(f"  Map data: {ann.getValue()}")

    elif isinstance(ann, omero.gateway.FileAnnotationWrapper):
        print(f"  File: {ann.getFile().getName()}")

    elif isinstance(ann, omero.gateway.CommentAnnotationWrapper):
        print(f"  Comment: {ann.getValue()}")

    print()
```

### 按命名空間篩選註解

```python
# 取得具有特定命名空間的註解
namespace = "mylab.qc.tags"

for ann in image.listAnnotations(ns=namespace):
    print(f"Found annotation: {ann.getId()}")

    if isinstance(ann, omero.gateway.MapAnnotationWrapper):
        for key, value in ann.getValue():
            print(f"  {key}: {value}")
```

### 取得具有命名空間的第一個註解

```python
# 透過命名空間取得單一註解
namespace = "mylab.analysis.results"
ann = dataset.getAnnotation(namespace)

if ann:
    print(f"Found annotation with namespace: {ann.getNs()}")
else:
    print("No annotation found with that namespace")
```

### 跨多個物件查詢註解

```python
# 取得連結到影像 ID 的所有標籤註解
image_ids = [1, 2, 3, 4, 5]

for link in conn.getAnnotationLinks('Image', parent_ids=image_ids):
    ann = link.getChild()

    if isinstance(ann._obj, omero.model.TagAnnotationI):
        print(f"Image {link.getParent().getId()}: Tag '{ann.getTextValue()}'")
```

## 計算註解

```python
# 計算專案上的註解
project_id = 123
count = conn.countAnnotations('Project', [project_id])
print(f"Project has {count[project_id]} annotations")

# 計算多張影像上的註解
image_ids = [1, 2, 3]
counts = conn.countAnnotations('Image', image_ids)

for image_id, count in counts.items():
    print(f"Image {image_id}: {count} annotations")
```

## 註解連結

### 手動建立註解連結

```python
# 取得註解和影像
tag = conn.getObject("TagAnnotation", tag_id)
image = conn.getObject("Image", image_id)

# 建立連結
link = omero.model.ImageAnnotationLinkI()
link.setParent(omero.model.ImageI(image.getId(), False))
link.setChild(omero.model.TagAnnotationI(tag.getId(), False))

# 儲存連結
conn.getUpdateService().saveAndReturnObject(link)
```

### 更新註解連結

```python
# 取得現有連結
annotation_ids = [1, 2, 3]
new_tag_id = 5

for link in conn.getAnnotationLinks('Image', ann_ids=annotation_ids):
    print(f"Image ID: {link.getParent().id}")

    # 變更連結的註解
    link._obj.child = omero.model.TagAnnotationI(new_tag_id, False)
    link.save()
```

## 移除註解

### 刪除註解

```python
# 取得影像
image = conn.getObject("Image", image_id)

# 收集要刪除的註解 ID
to_delete = []
namespace = "mylab.temp.annotations"

for ann in image.listAnnotations(ns=namespace):
    to_delete.append(ann.getId())

# 刪除註解
if to_delete:
    conn.deleteObjects('Annotation', to_delete, wait=True)
    print(f"Deleted {len(to_delete)} annotations")
```

### 取消連結註解（保留註解，移除連結）

```python
# 取得影像
image = conn.getObject("Image", image_id)

# 收集要刪除的連結 ID
to_delete = []

for ann in image.listAnnotations():
    if isinstance(ann, omero.gateway.TagAnnotationWrapper):
        to_delete.append(ann.link.getId())

# 刪除連結（註解保留在資料庫中）
if to_delete:
    conn.deleteObjects("ImageAnnotationLink", to_delete, wait=True)
    print(f"Unlinked {len(to_delete)} annotations")
```

### 刪除特定註解類型

```python
import omero.gateway

# 只刪除對應註解
image = conn.getObject("Image", image_id)
to_delete = []

for ann in image.listAnnotations():
    if isinstance(ann, omero.gateway.MapAnnotationWrapper):
        to_delete.append(ann.getId())

conn.deleteObjects('Annotation', to_delete, wait=True)
```

## 註解所有權

### 設定註解所有者（僅限管理員）

```python
import omero.model

# 建立具有特定所有者的標籤
tag_ann = omero.gateway.TagAnnotationWrapper(conn)
tag_ann.setValue("Admin Tag")

# 設定所有者（需要管理員權限）
user_id = 5
tag_ann._obj.details.owner = omero.model.ExperimenterI(user_id, False)
tag_ann.save()
```

### 以另一使用者建立註解（僅限管理員）

```python
# 管理員連線
admin_conn = BlitzGateway(admin_user, admin_pass, host=host, port=4064)
admin_conn.connect()

# 取得目標使用者
user_id = 10
user = admin_conn.getObject("Experimenter", user_id).getName()

# 建立該使用者的連線
user_conn = admin_conn.suConn(user)

# 以該使用者建立註解
map_ann = omero.gateway.MapAnnotationWrapper(user_conn)
map_ann.setNs("mylab.metadata")
map_ann.setValue([["key", "value"]])
map_ann.save()

# 連結到專案
project = admin_conn.getObject("Project", project_id)
project.linkAnnotation(map_ann)

# 關閉連線
user_conn.close()
admin_conn.close()
```

## 批次註解操作

### 為多張影像加標籤

```python
# 建立或取得標籤
tag = omero.gateway.TagAnnotationWrapper(conn)
tag.setValue("Validated")
tag.save()

# 取得要加標籤的影像
dataset = conn.getObject("Dataset", dataset_id)

# 為資料集中的所有影像加標籤
for image in dataset.listChildren():
    image.linkAnnotation(tag)
    print(f"Tagged image: {image.getName()}")
```

### 批次新增對應註解

```python
# 為多張影像準備中繼資料
image_metadata = {
    101: [["Quality", "Good"], ["Reviewed", "Yes"]],
    102: [["Quality", "Excellent"], ["Reviewed", "Yes"]],
    103: [["Quality", "Poor"], ["Reviewed", "No"]]
}

# 新增註解
for image_id, kv_data in image_metadata.items():
    image = conn.getObject("Image", image_id)

    if image:
        map_ann = omero.gateway.MapAnnotationWrapper(conn)
        map_ann.setNs("mylab.qc")
        map_ann.setValue(kv_data)
        map_ann.save()

        image.linkAnnotation(map_ann)
        print(f"Annotated image {image_id}")
```

## 命名空間

### 標準 OMERO 命名空間

```python
import omero.constants.metadata as omero_ns

# 用戶端對應註解命名空間
omero_ns.NSCLIENTMAPANNOTATION
# "openmicroscopy.org/omero/client/mapAnnotation"

# 批量註解命名空間
omero_ns.NSBULKANNOTATIONS
# "openmicroscopy.org/omero/bulk_annotations"
```

### 自訂命名空間

自訂命名空間的最佳實務：
- 使用反向網域名稱：`"org.mylab.category.subcategory"`
- 具體明確：`"com.company.project.analysis.v1"`
- 如果架構可能變更則包含版本：`"mylab.metadata.v2"`

```python
# 定義命名空間
NS_QC = "org.mylab.quality_control"
NS_ANALYSIS = "org.mylab.image_analysis.v1"
NS_PUBLICATION = "org.mylab.publication.2024"

# 在註解中使用
map_ann.setNs(NS_ANALYSIS)
```

## 按類型載入所有註解

### 載入所有檔案註解

```python
# 定義要包含/排除的命名空間
ns_to_include = ["mylab.analysis.results"]
ns_to_exclude = []

# 取得中繼資料服務
metadataService = conn.getMetadataService()

# 載入具有命名空間的所有檔案註解
annotations = metadataService.loadSpecifiedAnnotations(
    'omero.model.FileAnnotation',
    ns_to_include,
    ns_to_exclude,
    None
)

for ann in annotations:
    print(f"File Annotation ID: {ann.getId().getValue()}")
    print(f"  File: {ann.getFile().getName().getValue()}")
    print(f"  Size: {ann.getFile().getSize().getValue()} bytes")
```

## 完整範例

```python
from omero.gateway import BlitzGateway
import omero.gateway
import omero.constants.metadata

HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 取得資料集
    dataset = conn.getObject("Dataset", dataset_id)

    # 新增標籤
    tag = omero.gateway.TagAnnotationWrapper(conn)
    tag.setValue("Analysis Complete")
    tag.save()
    dataset.linkAnnotation(tag)

    # 新增具有中繼資料的對應註解
    metadata = [
        ["Analysis Date", "2024-10-20"],
        ["Software", "CellProfiler 4.2"],
        ["Pipeline", "cell_segmentation_v3"]
    ]
    map_ann = omero.gateway.MapAnnotationWrapper(conn)
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(metadata)
    map_ann.save()
    dataset.linkAnnotation(map_ann)

    # 新增檔案註解
    file_ann = conn.createFileAnnfromLocalFile(
        "analysis_summary.pdf",
        mimetype="application/pdf",
        ns="mylab.reports",
        desc="Analysis summary report"
    )
    dataset.linkAnnotation(file_ann)

    # 新增評論
    comment = omero.gateway.CommentAnnotationWrapper(conn)
    comment.setValue("Dataset ready for review")
    comment.save()
    dataset.linkAnnotation(comment)

    print(f"Added 4 annotations to dataset {dataset.getName()}")
```

## 最佳實務

1. **使用命名空間**：務必使用命名空間來組織註解
2. **描述性標籤**：使用清晰、一致的標籤名稱
3. **結構化中繼資料**：對結構化資料優先使用對應註解而非評論
4. **檔案組織**：使用描述性檔案名稱和 MIME 類型
5. **連結重用**：重用現有標籤而非建立重複項
6. **批次操作**：在迴圈中處理多個物件以提高效率
7. **錯誤處理**：在連結前檢查儲存是否成功
8. **清理**：不再需要時移除臨時註解
9. **文件記錄**：記錄自訂命名空間的含義
10. **權限**：考慮協作工作流程中的註解所有權
