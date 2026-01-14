# ZINC 資料庫 API 參考

## 概述

用於程式化存取 ZINC 資料庫的完整技術參考，涵蓋 API 端點、查詢語法、參數、回應格式，以及 ZINC22、ZINC20 和舊版本的進階使用模式。

## 基礎 URL

### ZINC22（目前版本）
- **CartBlanche22 API**：`https://cartblanche22.docking.org/`
- **檔案儲存庫**：`https://files.docking.org/zinc22/`
- **主網站**：`https://zinc.docking.org/`

### ZINC20（維護中）
- **API**：`https://zinc20.docking.org/`
- **檔案儲存庫**：`https://files.docking.org/zinc20/`

### 文件
- **Wiki**：`https://wiki.docking.org/`
- **GitHub**：`https://github.com/docking-org/`

## API 端點

### 1. 透過 ZINC ID 擷取物質

使用 ZINC 識別碼擷取化合物資訊。

**端點**：`/substances.txt`

**參數**：
- `zinc_id`（必填）：單一 ZINC ID 或逗號分隔列表
- `output_fields`（選填）：逗號分隔的欄位名稱（預設：所有欄位）

**URL 格式**：
```
https://cartblanche22.docking.org/substances.txt:zinc_id={ZINC_ID}&output_fields={FIELDS}
```

**範例**：

單一化合物：
```bash
curl "https://cartblanche22.docking.org/[email protected]_fields=zinc_id,smiles,catalogs"
```

多個化合物：
```bash
curl "https://cartblanche22.docking.org/substances.txt:zinc_id=ZINC000000000001,ZINC000000000002,ZINC000000000003&output_fields=zinc_id,smiles,tranche"
```

從檔案批次擷取：
```bash
# 建立包含 ZINC ID 的檔案（每行一個或逗號分隔）
curl -X POST "https://cartblanche22.docking.org/substances.txt?output_fields=zinc_id,smiles" \
  -F "zinc_id=@zinc_ids.txt"
```

**回應格式**（TSV）：
```
zinc_id	smiles	catalogs
ZINC000000000001	CC(C)O	[vendor1,vendor2]
ZINC000000000002	c1ccccc1	[vendor3]
```

### 2. 透過 SMILES 結構搜尋

透過化學結構搜尋化合物，可選擇相似度閾值。

**端點**：`/smiles.txt`

**參數**：
- `smiles`（必填）：查詢 SMILES 字串（必要時需 URL 編碼）
- `dist`（選填）：Tanimoto 距離閾值（0-10，預設：0 = 精確）
- `adist`（選填）：替代距離指標（0-10，預設：0）
- `output_fields`（選填）：逗號分隔的欄位名稱

**URL 格式**：
```
https://cartblanche22.docking.org/smiles.txt:smiles={SMILES}&dist={DIST}&adist={ADIST}&output_fields={FIELDS}
```

**範例**：

精確結構匹配：
```bash
curl "https://cartblanche22.docking.org/smiles.txt:smiles=c1ccccc1&output_fields=zinc_id,smiles"
```

相似性搜尋（Tanimoto 距離 = 3）：
```bash
curl "https://cartblanche22.docking.org/smiles.txt:smiles=CC(C)Cc1ccc(cc1)C(C)C(=O)O&dist=3&output_fields=zinc_id,smiles,catalogs"
```

廣泛相似性搜尋：
```bash
curl "https://cartblanche22.docking.org/smiles.txt:smiles=c1ccccc1&dist=5&adist=5&output_fields=zinc_id,smiles,tranche"
```

URL 編碼的 SMILES（用於特殊字元）：
```bash
# 原始：CC(=O)Oc1ccccc1C(=O)O
# 編碼：CC%28%3DO%29Oc1ccccc1C%28%3DO%29O
curl "https://cartblanche22.docking.org/smiles.txt:smiles=CC%28%3DO%29Oc1ccccc1C%28%3DO%29O&dist=2"
```

**距離參數解釋**：
- `dist=0`：精確匹配
- `dist=1-3`：近似類似物（高相似度）
- `dist=4-6`：中等類似物
- `dist=7-10`：多樣化學空間

### 3. 供應商代碼搜尋

透過供應商目錄編號查詢化合物。

**端點**：`/catitems.txt`

**參數**：
- `catitem_id`（必填）：供應商目錄代碼
- `output_fields`（選填）：逗號分隔的欄位名稱

**URL 格式**：
```
https://cartblanche22.docking.org/catitems.txt:catitem_id={SUPPLIER_CODE}&output_fields={FIELDS}
```

**範例**：
```bash
curl "https://cartblanche22.docking.org/catitems.txt:catitem_id=SUPPLIER-12345&output_fields=zinc_id,smiles,supplier_code,catalogs"
```

### 4. 隨機化合物取樣

產生隨機化合物集，可選擇依化學性質過濾。

**端點**：`/substance/random.txt`

**參數**：
- `count`（選填）：擷取的化合物數量（預設：100，最大值依伺服器而定）
- `subset`（選填）：依預定義子集過濾（例如 'lead-like'、'drug-like'、'fragment'）
- `output_fields`（選填）：逗號分隔的欄位名稱

**URL 格式**：
```
https://cartblanche22.docking.org/substance/random.txt:count={COUNT}&subset={SUBSET}&output_fields={FIELDS}
```

**範例**：

隨機 100 個化合物（預設）：
```bash
curl "https://cartblanche22.docking.org/substance/random.txt"
```

隨機類先導化合物分子：
```bash
curl "https://cartblanche22.docking.org/substance/random.txt:count=1000&subset=lead-like&output_fields=zinc_id,smiles,tranche"
```

隨機類藥物分子：
```bash
curl "https://cartblanche22.docking.org/substance/random.txt:count=5000&subset=drug-like&output_fields=zinc_id,smiles"
```

隨機片段：
```bash
curl "https://cartblanche22.docking.org/substance/random.txt:count=500&subset=fragment&output_fields=zinc_id,smiles,tranche"
```

**子集定義**：
- `fragment`：MW < 250，適用於基於片段的藥物發現
- `lead-like`：MW 250-350，LogP ≤ 3.5，可旋轉鍵 ≤ 7
- `drug-like`：MW 350-500，符合 Lipinski 五規則
- `lugs`：大型、異常優良子集（高度篩選）

## 輸出欄位

### 可用欄位

使用 `output_fields` 參數自訂 API 回應：

| 欄位 | 描述 | 範例 |
|-------|-------------|---------|
| `zinc_id` | ZINC 識別碼 | ZINC000000000001 |
| `smiles` | 標準 SMILES 字串 | CC(C)O |
| `sub_id` | 內部物質 ID | 123456 |
| `supplier_code` | 供應商目錄編號 | AB-1234567 |
| `catalogs` | 供應商列表 | [emolecules, mcule, mcule-ultimate] |
| `tranche` | 編碼的分子性質 | H02P025M300-0 |
| `mwt` | 分子量 | 325.45 |
| `logp` | LogP（分配係數） | 2.5 |
| `hba` | 氫鍵受體數 | 4 |
| `hbd` | 氫鍵供體數 | 2 |
| `rotatable_bonds` | 可旋轉鍵數 | 5 |

**注意**：並非所有欄位對所有端點都可用。欄位可用性取決於資料庫版本和端點。

### 預設欄位

如果未指定 `output_fields`，端點以 TSV 格式回傳所有可用欄位。

### 自訂欄位選擇

僅請求特定欄位：
```bash
curl "https://cartblanche22.docking.org/[email protected]_fields=zinc_id,smiles"
```

請求多個欄位：
```bash
curl "https://cartblanche22.docking.org/[email protected]_fields=zinc_id,smiles,tranche,catalogs"
```

## Tranche 系統

ZINC 根據分子性質將化合物組織成 tranches，以便有效過濾和組織。

### Tranche 代碼格式

**模式**：`H##P###M###-phase`

| 組成 | 描述 | 範圍 |
|-----------|-------------|-------|
| H## | 氫鍵供體數 | 00-99 |
| P### | LogP × 10 | 000-999（例如 P035 = LogP 3.5） |
| M### | 分子量 | 000-999 Da |
| phase | 反應性分類 | 0-9 |

### 範例

| Tranche 代碼 | 解釋 |
|--------------|----------------|
| `H00P010M250-0` | 0 個 H 供體，LogP=1.0，MW=250 Da，phase 0 |
| `H05P035M400-0` | 5 個 H 供體，LogP=3.5，MW=400 Da，phase 0 |
| `H02P-005M180-0` | 2 個 H 供體，LogP=-0.5，MW=180 Da，phase 0 |

### 反應性 Phases

| Phase | 描述 |
|-------|-------------|
| 0 | 無反應性（篩選首選） |
| 1-9 | 反應性遞增（PAINS、反應性基團） |

### 在 Python 中解析 Tranches

```python
import re

def parse_tranche(tranche_str):
    """
    解析 ZINC tranche 代碼。

    Args:
        tranche_str: Tranche 代碼（例如 "H05P035M400-0"）

    Returns:
        包含 h_donors、logp、mw、phase 的字典
    """
    pattern = r'H(\d+)P(-?\d+)M(\d+)-(\d+)'
    match = re.match(pattern, tranche_str)

    if not match:
        return None

    return {
        'h_donors': int(match.group(1)),
        'logp': int(match.group(2)) / 10.0,
        'mw': int(match.group(3)),
        'phase': int(match.group(4))
    }

# 使用範例
tranche = "H05P035M400-0"
props = parse_tranche(tranche)
print(props)  # {'h_donors': 5, 'logp': 3.5, 'mw': 400, 'phase': 0}
```

### 依 Tranches 過濾

從檔案儲存庫下載特定 tranches：
```bash
# 下載特定 tranche 中的所有化合物
wget https://files.docking.org/zinc22/H05/H05P035M400-0.db2.gz
```

## 檔案儲存庫存取

### 目錄結構

ZINC22 3D 結構依氫鍵供體數階層組織：

```
https://files.docking.org/zinc22/
├── H00/
│   ├── H00P010M200-0.db2.gz
│   ├── H00P020M250-0.db2.gz
│   └── ...
├── H01/
├── H02/
└── ...
```

### 檔案格式

| 副檔名 | 格式 | 描述 |
|-----------|--------|-------------|
| `.db2.gz` | DOCK 資料庫 | 用於 DOCK 的壓縮多構象 DB |
| `.mol2.gz` | MOL2 | 具有 3D 座標的多分子格式 |
| `.sdf.gz` | SDF | Structure-Data File 格式 |
| `.smi` | SMILES | 具有 ZINC ID 的純文字 SMILES |

### 下載 3D 結構

**單一 tranche**：
```bash
wget https://files.docking.org/zinc22/H05/H05P035M400-0.db2.gz
```

**多個 tranches**（使用 aria2c 平行下載）：
```bash
# 建立 URL 列表
cat > tranche_urls.txt <<EOF
https://files.docking.org/zinc22/H05/H05P035M400-0.db2.gz
https://files.docking.org/zinc22/H05/H05P035M400-0.db2.gz
https://files.docking.org/zinc22/H05/H05P040M400-0.db2.gz
EOF

# 平行下載
aria2c -i tranche_urls.txt -x 8 -j 4
```

**遞迴下載**（謹慎使用 - 資料量大）：
```bash
wget -r -np -nH --cut-dirs=1 -A "*.db2.gz" \
  https://files.docking.org/zinc22/H05/
```

### 解壓縮結構

```bash
# 解壓縮
gunzip H05P035M400-0.db2.gz

# 使用 OpenBabel 轉換為其他格式
obabel H05P035M400-0.db2 -O output.sdf
obabel H05P035M400-0.db2 -O output.mol2
```

## 進階查詢模式

### 結合多個搜尋條件

**用於複雜查詢的 Python 包裝器**：

```python
import subprocess
import pandas as pd
from io import StringIO

def advanced_zinc_search(smiles=None, zinc_ids=None, dist=0,
                         subset=None, count=None, output_fields=None):
    """
    具有多個條件的彈性 ZINC 搜尋。

    Args:
        smiles: 用於結構搜尋的 SMILES 字串
        zinc_ids: 用於批次擷取的 ZINC ID 列表
        dist: 相似性的距離參數（0-10）
        subset: 子集過濾（lead-like、drug-like、fragment）
        count: 隨機化合物數量
        output_fields: 要回傳的欄位列表

    Returns:
        包含結果的 pandas DataFrame
    """
    if output_fields is None:
        output_fields = ['zinc_id', 'smiles', 'tranche', 'catalogs']

    fields_str = ','.join(output_fields)

    # 結構搜尋
    if smiles:
        url = f"https://cartblanche22.docking.org/smiles.txt:smiles={smiles}&dist={dist}&output_fields={fields_str}"

    # 批次擷取
    elif zinc_ids:
        zinc_ids_str = ','.join(zinc_ids)
        url = f"https://cartblanche22.docking.org/substances.txt:zinc_id={zinc_ids_str}&output_fields={fields_str}"

    # 隨機取樣
    elif count:
        url = f"https://cartblanche22.docking.org/substance/random.txt:count={count}&output_fields={fields_str}"
        if subset:
            url += f"&subset={subset}"

    else:
        raise ValueError("必須指定 smiles、zinc_ids 或 count")

    # 執行查詢
    result = subprocess.run(['curl', '-s', url],
                          capture_output=True, text=True)

    # 解析為 DataFrame
    df = pd.read_csv(StringIO(result.stdout), sep='\t')

    return df
```

**使用範例**：

```python
# 尋找相似化合物
df = advanced_zinc_search(
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    dist=3,
    output_fields=['zinc_id', 'smiles', 'catalogs']
)

# 批次擷取
zinc_ids = ["ZINC000000000001", "ZINC000000000002"]
df = advanced_zinc_search(zinc_ids=zinc_ids)

# 隨機類藥物集
df = advanced_zinc_search(
    count=1000,
    subset='drug-like',
    output_fields=['zinc_id', 'smiles', 'tranche']
)
```

### 基於性質的過濾

使用 tranche 資料依分子性質過濾化合物：

```python
def filter_by_properties(df, mw_range=None, logp_range=None,
                        max_hbd=None, phase=0):
    """
    依分子性質過濾 DataFrame。

    Args:
        df: 具有 'tranche' 欄位的 DataFrame
        mw_range: 元組 (min_mw, max_mw)
        logp_range: 元組 (min_logp, max_logp)
        max_hbd: 最大氫鍵供體數
        phase: 反應性 phase（0 = 無反應性）

    Returns:
        過濾後的 DataFrame
    """
    # 解析 tranches
    df['tranche_props'] = df['tranche'].apply(parse_tranche)
    df['mw'] = df['tranche_props'].apply(lambda x: x['mw'] if x else None)
    df['logp'] = df['tranche_props'].apply(lambda x: x['logp'] if x else None)
    df['hbd'] = df['tranche_props'].apply(lambda x: x['h_donors'] if x else None)
    df['phase'] = df['tranche_props'].apply(lambda x: x['phase'] if x else None)

    # 套用過濾器
    mask = pd.Series([True] * len(df))

    if mw_range:
        mask &= (df['mw'] >= mw_range[0]) & (df['mw'] <= mw_range[1])

    if logp_range:
        mask &= (df['logp'] >= logp_range[0]) & (df['logp'] <= logp_range[1])

    if max_hbd is not None:
        mask &= df['hbd'] <= max_hbd

    if phase is not None:
        mask &= df['phase'] == phase

    return df[mask]

# 範例：取得具有特定性質的類藥物化合物
df = advanced_zinc_search(count=10000, subset='drug-like')
filtered = filter_by_properties(
    df,
    mw_range=(300, 450),
    logp_range=(1.0, 4.0),
    max_hbd=3,
    phase=0
)
```

## 速率限制與最佳實踐

### 速率限制

ZINC 未公布明確的速率限制，但使用者應：

- **避免快速連續請求**：查詢間隔至少 1 秒
- **使用批次操作**：在單一請求中查詢多個 ZINC ID
- **快取結果**：將經常存取的資料儲存在本地
- **離峰使用**：在離峰時段進行大量下載（UTC 夜間/週末）

### 禮儀

```python
import time

def polite_zinc_query(query_func, *args, delay=1.0, **kwargs):
    """在查詢間新增延遲的包裝器。"""
    result = query_func(*args, **kwargs)
    time.sleep(delay)
    return result
```

### 錯誤處理

```python
def robust_zinc_query(url, max_retries=3, timeout=30):
    """
    具有重試邏輯的 ZINC 查詢。

    Args:
        url: 完整的 ZINC API URL
        max_retries: 最大重試次數
        timeout: 請求逾時秒數

    Returns:
        查詢結果或失敗時回傳 None
    """
    import subprocess
    import time

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['curl', '-s', '--max-time', str(timeout), url],
                capture_output=True,
                text=True,
                check=True
            )

            # 檢查空的或錯誤回應
            if not result.stdout or 'error' in result.stdout.lower():
                raise ValueError("無效的回應")

            return result.stdout

        except (subprocess.CalledProcessError, ValueError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指數退避
                print(f"重試 {attempt + 1}/{max_retries}，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"在 {max_retries} 次嘗試後失敗")
                return None
```

## 與分子對接整合

### 準備 DOCK6 庫

```bash
# 1. 下載 tranche 檔案
wget https://files.docking.org/zinc22/H05/H05P035M400-0.db2.gz

# 2. 解壓縮
gunzip H05P035M400-0.db2.gz

# 3. 直接與 DOCK6 使用
dock6 -i dock.in -o dock.out -l H05P035M400-0.db2
```

### AutoDock Vina 整合

```bash
# 1. 下載 MOL2 格式
wget https://files.docking.org/zinc22/H05/H05P035M400-0.mol2.gz
gunzip H05P035M400-0.mol2.gz

# 2. 使用 prepare_ligand 腳本轉換為 PDBQT
prepare_ligand4.py -l H05P035M400-0.mol2 -o ligands.pdbqt -A hydrogens

# 3. 執行 Vina
vina --receptor protein.pdbqt --ligand ligands.pdbqt \
     --center_x 25.0 --center_y 25.0 --center_z 25.0 \
     --size_x 20.0 --size_y 20.0 --size_z 20.0
```

### RDKit 整合

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd

def process_zinc_results(zinc_df):
    """
    使用 RDKit 處理 ZINC 結果。

    Args:
        zinc_df: 具有 SMILES 欄位的 DataFrame

    Returns:
        具有計算性質的 DataFrame
    """
    # 將 SMILES 轉換為分子
    zinc_df['mol'] = zinc_df['smiles'].apply(Chem.MolFromSmiles)

    # 計算性質
    zinc_df['mw'] = zinc_df['mol'].apply(Descriptors.MolWt)
    zinc_df['logp'] = zinc_df['mol'].apply(Descriptors.MolLogP)
    zinc_df['hbd'] = zinc_df['mol'].apply(Descriptors.NumHDonors)
    zinc_df['hba'] = zinc_df['mol'].apply(Descriptors.NumHAcceptors)
    zinc_df['tpsa'] = zinc_df['mol'].apply(Descriptors.TPSA)
    zinc_df['rotatable'] = zinc_df['mol'].apply(Descriptors.NumRotatableBonds)

    # 產生 3D 構象
    for mol in zinc_df['mol']:
        if mol:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

    return zinc_df

# 儲存為 SDF 以供對接
def save_to_sdf(zinc_df, output_file):
    """將分子儲存為 SDF 檔案。"""
    writer = Chem.SDWriter(output_file)
    for idx, row in zinc_df.iterrows():
        if row['mol']:
            row['mol'].SetProp('ZINC_ID', row['zinc_id'])
            writer.write(row['mol'])
    writer.close()
```

## 疑難排解

### 常見問題

**問題**：空的或沒有結果
- **解決方案**：檢查 SMILES 語法，驗證 ZINC ID 存在，嘗試更廣泛的相似性搜尋

**問題**：逾時錯誤
- **解決方案**：減少結果數量，使用批次查詢，在離峰時段嘗試

**問題**：無效的 SMILES 編碼
- **解決方案**：URL 編碼特殊字元（在 Python 中使用 `urllib.parse.quote()`）

**問題**：找不到 Tranche 檔案
- **解決方案**：驗證 tranche 代碼格式，檢查檔案儲存庫結構

### 除錯模式

```python
def debug_zinc_query(url):
    """列印查詢詳細資訊以進行除錯。"""
    print(f"查詢 URL：{url}")

    result = subprocess.run(['curl', '-v', url],
                          capture_output=True, text=True)

    print(f"狀態：{result.returncode}")
    print(f"Stderr：{result.stderr}")
    print(f"Stdout 長度：{len(result.stdout)}")
    print(f"前 500 字元：\n{result.stdout[:500]}")

    return result.stdout
```

## 版本差異

### ZINC22 vs ZINC20 vs ZINC15

| 功能 | ZINC22 | ZINC20 | ZINC15 |
|---------|--------|--------|--------|
| 化合物數量 | 2.3 億+ 可購買 | 專注於先導化合物 | 總計約 7.5 億 |
| API | CartBlanche22 | 類似 | REST-like |
| Tranches | 有 | 有 | 有 |
| 3D 結構 | 有 | 有 | 有 |
| 狀態 | 目前，持續增長 | 維護中 | 舊版 |

### API 相容性

大多數查詢模式跨版本通用，但 URL 不同：
- ZINC22：`cartblanche22.docking.org`
- ZINC20：`zinc20.docking.org`
- ZINC15：`zinc15.docking.org`

## 其他資源

- **ZINC Wiki**：https://wiki.docking.org/
- **ZINC22 文件**：https://wiki.docking.org/index.php/Category:ZINC22
- **ZINC API 指南**：https://wiki.docking.org/index.php/ZINC_api
- **檔案存取指南**：https://wiki.docking.org/index.php/ZINC22:Getting_started
- **出版物**：
  - ZINC22：J. Chem. Inf. Model. 2023
  - ZINC15：J. Chem. Inf. Model. 2020, 60, 6065-6073
- **支援**：透過 ZINC 網站或 GitHub issues 聯繫

