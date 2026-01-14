# BRENDA 資料庫 API 參考

## 概述

本文件提供 BRENDA（BRaunschweig ENzyme DAtabase）SOAP API 和 Python 用戶端實作的詳細參考資訊。BRENDA 是全球最完整的酵素資訊系統，包含超過 45,000 種酵素及數百萬筆動力學資料點。

## SOAP API 端點

### 基礎 WSDL URL
```
https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl
```

### 驗證

所有 BRENDA API 呼叫都需要使用電子郵件和密碼進行驗證：

**參數：**
- `email`: 您註冊的 BRENDA 電子郵件地址
- `password`: 您的 BRENDA 帳戶密碼

**驗證流程：**
1. 密碼在傳輸前使用 SHA-256 進行雜湊處理
2. 電子郵件和雜湊密碼作為每個 API 呼叫的前兩個參數
3. 支援舊版 `BRENDA_EMIAL` 環境變數（注意拼寫錯誤）

## 可用的 SOAP 動作

### getKmValue

擷取酵素的 Michaelis 常數（Km）值。

**參數：**
1. `email`: BRENDA 帳戶電子郵件
2. `passwordHash`: SHA-256 雜湊密碼
3. `ecNumber*`: 酵素的 EC 編號（允許萬用字元）
4. `organism*`: 生物體名稱（允許萬用字元，預設："*"）
5. `kmValue*`: Km 值欄位（預設："*"）
6. `kmValueMaximum*`: 最大 Km 值欄位（預設："*"）
7. `substrate*`: 底物名稱（允許萬用字元，預設："*"）
8. `commentary*`: 註解欄位（預設："*"）
9. `ligandStructureId*`: 配體結構 ID 欄位（預設："*"）
10. `literature*`: 文獻參考欄位（預設："*"）

**萬用字元：**
- `*`: 匹配任何序列
- 可與部分 EC 編號一起使用（例如 "1.1.*"）

**回應格式：**
```
organism*Escherichia coli#substrate*glucose#kmValue*0.12#kmValueMaximum*#commentary*pH 7.4, 25°C#ligandStructureId*#literature*
```

**回應欄位範例：**
- `organism`: 來源生物體
- `substrate`: 底物名稱
- `kmValue`: Michaelis 常數值（通常以 mM 為單位）
- `kmValueMaximum`: 最大 Km 值（如果有的話）
- `commentary`: 實驗條件（pH、溫度等）
- `ligandStructureId`: BRENDA 配體結構識別碼
- `literature`: 主要文獻參考

### getReaction

擷取酵素的反應方程式和化學計量。

**參數：**
1. `email`: BRENDA 帳戶電子郵件
2. `passwordHash`: SHA-256 雜湊密碼
3. `ecNumber*`: 酵素的 EC 編號（允許萬用字元）
4. `organism*`: 生物體名稱（允許萬用字元，預設："*"）
5. `reaction*`: 反應方程式（允許萬用字元，預設："*"）
6. `commentary*`: 註解欄位（預設："*"）
7. `literature*`: 文獻參考欄位（預設："*"）

**回應格式：**
```
ecNumber*1.1.1.1#organism*Saccharomyces cerevisiae#reaction*ethanol + NAD+ <=> acetaldehyde + NADH + H+#commentary*#literature*
```

**回應欄位範例：**
- `ecNumber`: 酵素委員會編號
- `organism`: 來源生物體
- `reaction`: 平衡化學方程式（使用 <=> 表示平衡，-> 表示方向）
- `commentary`: 額外資訊
- `literature`: 參考引文

## 資料欄位規範

### EC 編號格式

EC 編號遵循標準階層格式：`A.B.C.D`

- **A**: 主類別（1-6）
  - 1: 氧化還原酶
  - 2: 轉移酶
  - 3: 水解酶
  - 4: 裂解酶
  - 5: 異構酶
  - 6: 連接酶
- **B**: 子類別
- **C**: 次子類別
- **D**: 序列號

**範例：**
- `1.1.1.1`: 醇脫氫酶
- `1.1.1.2`: 醇脫氫酶（NADP+）
- `3.2.1.23`: β-半乳糖苷酶
- `2.7.1.1`: 己糖激酶

### 生物體名稱

生物體名稱應使用正確的二名法：

**正確格式：**
- `Escherichia coli`
- `Saccharomyces cerevisiae`
- `Homo sapiens`

**萬用字元：**
- `Escherichia*`: 匹配所有大腸桿菌菌株
- `*coli`: 匹配所有 coli 物種
- `*`: 匹配所有生物體

### 底物名稱

底物名稱遵循 IUPAC 或常見生化慣例：

**常見格式：**
- 化學名稱：`glucose`、`ethanol`、`pyruvate`
- IUPAC 名稱：`β-D-glucose`、`ethanol`、`2-oxopropanoic acid`
- 縮寫：`ATP`、`NAD+`、`CoA`

**特殊情況：**
- 輔因子：`NAD+`、`NADH`、`NADP+`、`NADPH`
- 金屬離子：`Mg2+`、`Zn2+`、`Fe2+`
- 無機化合物：`H2O`、`CO2`、`O2`

### 註解欄位格式

註解欄位包含實驗條件和其他元資料：

**常見資訊：**
- **pH**: `pH 7.4`、`pH 6.5-8.0`
- **溫度**: `25°C`、`37°C`、`50-60°C`
- **緩衝系統**: `phosphate buffer`、`Tris-HCl`
- **純度**: `purified enzyme`、`crude extract`
- **測定條件**: `spectrophotometric`、`radioactive`
- **抑制**: `inhibited by heavy metals`、`activated by Mg2+`

**範例：**
- `pH 7.4, 25°C, phosphate buffer`
- `pH 6.5-8.0 optimum, thermostable enzyme`
- `purified enzyme, specific activity 125 U/mg`
- `inhibited by iodoacetate, activated by Mn2+`

### 反應方程式格式

反應使用標準生化符號：

**符號：**
- `+`: 分隔反應物/產物
- `<=>`: 可逆反應
- `->`: 不可逆（定向）反應
- `=`: 反應的替代表示法

**常見模式：**
- **氧化/還原**: `alcohol + NAD+ <=> aldehyde + NADH + H+`
- **磷酸化**: `glucose + ATP <=> glucose-6-phosphate + ADP`
- **水解**: `ester + H2O <=> acid + alcohol`
- **羧化**: `acetyl-CoA + CO2 + H2O <=> malonyl-CoA`

**輔因子需求：**
- **氧化還原酶**: NAD+、NADH、NADP+、NADPH、FAD、FADH2
- **轉移酶**: ATP、ADP、GTP、GDP
- **連接酶**: ATP、CoA

## 速率限制和使用

### API 速率限制

- **最大**: 每秒 5 個請求
- **持續**: 建議每秒 1 個請求
- **每日配額**: 因帳戶類型而異

### 最佳實務

1. **實作延遲**: 在請求之間加入 0.5-1 秒延遲
2. **快取結果**: 在本地儲存經常存取的資料
3. **使用特定搜尋**: 盡可能依生物體和底物縮小範圍
4. **批次操作**: 將相關查詢分組
5. **優雅地處理錯誤**: 檢查 HTTP 和 SOAP 錯誤
6. **謹慎使用萬用字元**: 廣泛搜尋會回傳大型資料集

### 錯誤處理

**常見 SOAP 錯誤：**
- `Authentication failed`: 檢查電子郵件/密碼
- `No data found`: 驗證 EC 編號、生物體、底物拼寫
- `Rate limit exceeded`: 降低請求頻率
- `Invalid parameters`: 檢查參數格式和順序

**網路錯誤：**
- 連線逾時
- SSL/TLS 錯誤
- 服務無法使用

## Python 用戶端參考

### brenda_client 模組

#### 核心函數

**`load_env_from_file(path=".env")`**
- **用途**: 從 .env 檔案載入環境變數
- **參數**: `path` - .env 檔案路徑（預設：".env"）
- **回傳**: None（填充 os.environ）

**`_get_credentials() -> tuple[str, str]`**
- **用途**: 從環境擷取 BRENDA 憑證
- **回傳**: (email, password) 元組
- **例外**: 若憑證遺失則引發 RuntimeError

**`_get_client() -> Client`**
- **用途**: 初始化或擷取 SOAP 用戶端
- **回傳**: Zeep Client 實例
- **特性**: 單例模式，自訂傳輸設定

**`_hash_password(password: str) -> str`**
- **用途**: 產生密碼的 SHA-256 雜湊
- **參數**: `password` - 明文密碼
- **回傳**: 十六進位 SHA-256 雜湊

**`call_brenda(action: str, parameters: List[str]) -> str`**
- **用途**: 執行 BRENDA SOAP 動作
- **參數**:
  - `action` - SOAP 動作名稱（例如 "getKmValue"）
  - `parameters` - 正確順序的參數清單
- **回傳**: 來自 BRENDA 的原始回應字串

#### 便利函數

**`get_km_values(ec_number: str, organism: str = "*", substrate: str = "*") -> List[str]`**
- **用途**: 擷取指定酵素的 Km 值
- **參數**:
  - `ec_number`: 酵素委員會編號
  - `organism`: 生物體名稱（允許萬用字元，預設："*"）
  - `substrate`: 底物名稱（允許萬用字元，預設："*"）
- **回傳**: 解析後的資料字串清單

**`get_reactions(ec_number: str, organism: str = "*", reaction: str = "*") -> List[str]`**
- **用途**: 擷取指定酵素的反應資料
- **參數**:
  - `ec_number`: 酵素委員會編號
  - `organism`: 生物體名稱（允許萬用字元，預設："*"）
  - `reaction`: 反應模式（允許萬用字元，預設："*"）
- **回傳**: 反應資料字串清單

#### 公用函數

**`split_entries(return_text: str) -> List[str]`**
- **用途**: 將 BRENDA 回應標準化為清單格式
- **參數**: `return_text` - 來自 BRENDA 的原始回應
- **回傳**: 個別資料項目清單
- **特性**: 處理字串和複雜物件回應

## 資料結構和解析

### Km 項目結構

**解析後的 Km 項目字典：**
```python
{
    'ecNumber': '1.1.1.1',
    'organism': 'Escherichia coli',
    'substrate': 'ethanol',
    'kmValue': '0.12',
    'km_value_numeric': 0.12,  # 擷取的數值
    'kmValueMaximum': '',
    'commentary': 'pH 7.4, 25°C',
    'ph': 7.4,               # 從註解擷取
    'temperature': 25.0,      # 從註解擷取
    'ligandStructureId': '',
    'literature': ''
}
```

### 反應項目結構

**解析後的反應項目字典：**
```python
{
    'ecNumber': '1.1.1.1',
    'organism': 'Saccharomyces cerevisiae',
    'reaction': 'ethanol + NAD+ <=> acetaldehyde + NADH + H+',
    'reactants': ['ethanol', 'NAD+'],
    'products': ['acetaldehyde', 'NADH', 'H+'],
    'commentary': '',
    'literature': ''
}
```

## 查詢模式和範例

### 基本查詢

**取得酵素的所有 Km 值：**
```python
from brenda_client import get_km_values

# 取得所有醇脫氫酶 Km 值
km_data = get_km_values("1.1.1.1")
```

**取得特定生物體的 Km 值：**
```python
# 取得人類醇脫氫酶 Km 值
human_km = get_km_values("1.1.1.1", organism="Homo sapiens")
```

**取得特定底物的 Km 值：**
```python
# 取得乙醇氧化的 Km
ethanol_km = get_km_values("1.1.1.1", substrate="ethanol")
```

### 萬用字元搜尋

**搜尋酵素家族：**
```python
# 所有醇脫氫酶
alcohol_dehydrogenases = get_km_values("1.1.1.*")

# 所有己糖激酶
hexokinases = get_km_values("2.7.1.*")
```

**搜尋生物體群組：**
```python
# 所有大腸桿菌菌株
e_coli_enzymes = get_km_values("*", organism="Escherichia coli")

# 所有芽孢桿菌屬
bacillus_enzymes = get_km_values("*", organism="Bacillus*")
```

### 組合搜尋

**特定酵素-底物組合：**
```python
# 取得酵母菌中葡萄糖氧化的 Km 值
glucose_km = get_km_values("1.1.1.1",
                          organism="Saccharomyces cerevisiae",
                          substrate="glucose")
```

### 反應查詢

**取得酵素的所有反應：**
```python
from brenda_client import get_reactions

reactions = get_reactions("1.1.1.1")
```

**搜尋含有特定底物的反應：**
```python
# 尋找涉及葡萄糖的反應
glucose_reactions = get_reactions("*", reaction="*glucose*")
```

## 資料分析模式

### 動力學參數分析

**擷取數值 Km 值：**
```python
from scripts.brenda_queries import parse_km_entry

km_data = get_km_values("1.1.1.1", substrate="ethanol")
numeric_kms = []

for entry in km_data:
    parsed = parse_km_entry(entry)
    if 'km_value_numeric' in parsed:
        numeric_kms.append(parsed['km_value_numeric'])

if numeric_kms:
    print(f"平均 Km: {sum(numeric_kms)/len(numeric_kms):.3f}")
    print(f"範圍: {min(numeric_kms):.3f} - {max(numeric_kms):.3f}")
```

### 生物體比較

**比較不同生物體間的酵素特性：**
```python
from scripts.brenda_queries import compare_across_organisms

organisms = ["Escherichia coli", "Saccharomyces cerevisiae", "Homo sapiens"]
comparison = compare_across_organisms("1.1.1.1", organisms)

for org_data in comparison:
    if org_data.get('data_points', 0) > 0:
        print(f"{org_data['organism']}: {org_data['average_km']:.3f}")
```

### 底物特異性

**分析底物偏好：**
```python
from scripts.brenda_queries import get_substrate_specificity

specificity = get_substrate_specificity("1.1.1.1")

for substrate_data in specificity[:5]:  # 前 5 個
    print(f"{substrate_data['name']}: Km = {substrate_data['km']:.3f}")
```

## 整合範例

### 代謝途徑建構

**建立酵素途徑：**
```python
from scripts.enzyme_pathway_builder import find_pathway_for_product

# 尋找乳酸生產途徑
pathway = find_pathway_for_product("lactate", max_steps=3)

for step in pathway['steps']:
    print(f"步驟 {step['step_number']}: {step['substrate']} -> {step['product']}")
    print(f"可用酵素: {len(step['enzymes'])}")
```

### 酵素工程支援

**尋找耐熱變異體：**
```python
from scripts.brenda_queries import find_thermophilic_homologs

thermophilic = find_thermophilic_homologs("1.1.1.1", min_temp=50)

for enzyme in thermophilic:
    print(f"{enzyme['organism']}: {enzyme['optimal_temperature']}°C")
```

### 動力學建模

**擷取建模參數：**
```python
from scripts.brenda_queries import get_modeling_parameters

model_data = get_modeling_parameters("1.1.1.1", substrate="ethanol")

print(f"Km: {model_data['km']}")
print(f"Vmax: {model_data['vmax']}")
print(f"最適條件: pH {model_data['ph']}, {model_data['temperature']}°C")
```

## 疑難排解

### 常見問題

**驗證錯誤：**
- 檢查 BRENDA_EMAIL 和 BRENDA_PASSWORD 環境變數
- 驗證帳戶已啟用且有 API 存取權限
- 注意舊版 BRENDA_EMIAL 支援（變數名稱拼寫錯誤）

**無回傳資料：**
- 驗證 EC 編號格式（例如 "1.1.1.1"，而非 "1.1.1"）
- 檢查生物體和底物名稱拼寫
- 嘗試使用萬用字元進行更廣泛的搜尋
- 某些酵素在 BRENDA 中可能資料有限

**速率限制：**
- 在請求之間實作延遲
- 在本地快取結果
- 使用更具體的查詢以減少資料量
- 考慮批次操作

**資料格式問題：**
- 使用提供的解析函數
- 優雅地處理缺失欄位
- BRENDA 資料格式可能不一致
- 使用前驗證解析的資料

### 效能優化

**查詢效率：**
- 已知時使用特定 EC 編號
- 依生物體或底物限制以減少結果大小
- 快取經常存取的資料
- 批次處理類似請求

**記憶體管理：**
- 分塊處理大型資料集
- 對大型結果集使用生成器
- 不再需要時清除解析的資料

**網路優化：**
- 對網路錯誤實作重試邏輯
- 使用適當的逾時設定
- 監控請求頻率

## 其他資源

### 官方文件

- **BRENDA 網站**: https://www.brenda-enzymes.org/
- **SOAP API 文件**: https://www.brenda-enzymes.org/soap.php
- **酵素命名法**: https://www.iubmb.org/enzyme/
- **EC 編號資料庫**: https://www.qmul.ac.uk/sbcs/iubmb/enzyme/

### 相關函式庫

- **Zeep（SOAP 用戶端）**: https://python-zeep.readthedocs.io/
- **PubChemPy**: https://pubchempy.readthedocs.io/
- **BioPython**: https://biopython.org/
- **RDKit**: https://www.rdkit.org/

### 資料格式

- **酵素委員會編號**: IUBMB 酵素分類
- **IUPAC 命名法**: 化學命名慣例
- **生化反應**: 標準方程式表示法
- **動力學參數**: Michaelis-Menten 動力學

### 社群資源

- **BRENDA 服務台**: 透過官方網站提供支援
- **生物資訊論壇**: Stack Overflow、Biostars
- **GitHub Issues**: 專案特定的錯誤報告
- **研究論文**: 酵素資料的主要文獻

---

*本 API 參考涵蓋 BRENDA SOAP API 和 Python 用戶端的核心功能。有關可用資料欄位和查詢模式的完整詳細資訊，請參閱官方 BRENDA 文件。*
