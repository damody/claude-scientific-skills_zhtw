# FDA 其他資料庫 - 物質與 NSDE

本參考文件涵蓋透過 openFDA 可存取的 FDA 物質相關和其他專業 API 端點。

## 概述

FDA 維護額外的資料庫，提供精確到分子層級的物質資訊。這些資料庫支援跨藥物、生物製劑、器材、食品和化妝品的法規活動。

## 可用端點

### 1. 物質資料

**端點**：`https://api.fda.gov/other/substance.json`

**目的**：存取精確到分子層級的物質資訊，供內部和外部使用。這包括活性藥物成分、賦形劑和用於 FDA 監管產品的其他物質的資訊。

**資料來源**：FDA 全球物質註冊系統（GSRS）

**主要欄位**：
- `uuid` - 唯一物質識別碼（UUID）
- `approvalID` - FDA 唯一成分識別碼（UNII）
- `approved` - 核准日期
- `substanceClass` - 物質類型（化學、蛋白質、核酸、聚合物等）
- `names` - 物質名稱陣列
- `names.name` - 名稱文字
- `names.type` - 名稱類型（系統、品牌、通用等）
- `names.preferred` - 是否為首選名稱
- `codes` - 物質代碼陣列
- `codes.code` - 代碼值
- `codes.codeSystem` - 代碼系統（CAS、ECHA、EINECS 等）
- `codes.type` - 代碼類型
- `relationships` - 物質關係陣列
- `relationships.type` - 關係類型（活性部分、代謝物、雜質等）
- `relationships.relatedSubstance` - 相關物質參考
- `moieties` - 分子部分
- `properties` - 理化性質陣列
- `properties.name` - 性質名稱
- `properties.value` - 性質值
- `properties.propertyType` - 性質類型
- `structure` - 化學結構資訊
- `structure.smiles` - SMILES 表示法
- `structure.inchi` - InChI 字串
- `structure.inchiKey` - InChI 鍵
- `structure.formula` - 分子式
- `structure.molecularWeight` - 分子量
- `modifications` - 結構修飾（用於蛋白質等）
- `protein` - 蛋白質特定資訊
- `protein.subunits` - 蛋白質亞基
- `protein.sequenceType` - 序列類型
- `nucleicAcid` - 核酸資訊
- `nucleicAcid.subunits` - 序列亞基
- `polymer` - 聚合物資訊
- `mixture` - 混合物成分
- `mixture.components` - 成分物質
- `tags` - 物質標籤
- `references` - 文獻參考

**物質類別**：
- **Chemical** - 具有定義化學結構的小分子
- **Protein** - 蛋白質和多肽
- **Nucleic Acid** - DNA、RNA、寡核苷酸
- **Polymer** - 聚合物質
- **Structurally Diverse** - 複雜混合物、植物提取物
- **Mixture** - 已定義的混合物
- **Concept** - 抽象概念（例如：群組）

**常見使用案例**：
- 活性成分識別
- 分子結構查詢
- UNII 代碼解析
- 化學識別碼對應（CAS 到 UNII 等）
- 物質關係分析
- 賦形劑識別
- 植物物質資訊
- 蛋白質和生物製劑特性描述

**查詢範例**：
```python
import requests

api_key = "YOUR_API_KEY"
url = "https://api.fda.gov/other/substance.json"

# 按 UNII 代碼查詢物質
params = {
    "api_key": api_key,
    "search": "approvalID:R16CO5Y76E",  # 阿斯匹靈 UNII
    "limit": 1
}

response = requests.get(url, params=params)
data = response.json()
```

```python
# 按物質名稱搜尋
params = {
    "api_key": api_key,
    "search": "names.name:acetaminophen",
    "limit": 5
}
```

```python
# 按 CAS 編號查詢物質
params = {
    "api_key": api_key,
    "search": "codes.code:50-78-2",  # 阿斯匹靈 CAS
    "limit": 1
}
```

```python
# 僅取得化學物質
params = {
    "api_key": api_key,
    "search": "substanceClass:chemical",
    "limit": 100
}
```

```python
# 按分子式搜尋
params = {
    "api_key": api_key,
    "search": "structure.formula:C8H9NO2",  # 對乙醯氨基酚
    "limit": 10
}
```

```python
# 查詢蛋白質物質
params = {
    "api_key": api_key,
    "search": "substanceClass:protein",
    "limit": 50
}
```

### 2. NSDE（國家物質資料庫條目）

**端點**：`https://api.fda.gov/other/nsde.json`

**目的**：存取舊版國家藥品代碼（NDC）目錄條目中的歷史物質資料。此端點提供歷史藥品列名中的物質資訊。

**注意**：此資料庫主要用於歷史參考。如需最新物質資訊，請使用物質資料端點。

**主要欄位**：
- `proprietary_name` - 產品專有名稱
- `nonproprietary_name` - 非專有名稱
- `dosage_form` - 劑型
- `route` - 給藥途徑
- `company_name` - 公司名稱
- `substance_name` - 物質名稱
- `active_numerator_strength` - 活性成分強度（分子）
- `active_ingred_unit` - 活性成分單位
- `pharm_classes` - 藥理類別
- `dea_schedule` - DEA 管制物質等級

**常見使用案例**：
- 歷史藥物配方研究
- 舊系統整合
- 歷史物質名稱對應
- 藥學歷史研究

**查詢範例**：
```python
# 按物質名稱搜尋
params = {
    "api_key": api_key,
    "search": "substance_name:ibuprofen",
    "limit": 20
}

response = requests.get("https://api.fda.gov/other/nsde.json", params=params)
```

```python
# 按 DEA 等級查詢管制物質
params = {
    "api_key": api_key,
    "search": "dea_schedule:CII",
    "limit": 50
}
```

## 整合技巧

### UNII 到 CAS 對應

```python
def get_substance_identifiers(unii, api_key):
    """
    取得給定 UNII 代碼的物質所有識別碼。

    參數：
        unii: FDA 唯一成分識別碼
        api_key: FDA API 金鑰

    回傳：
        包含物質識別碼的字典
    """
    import requests

    url = "https://api.fda.gov/other/substance.json"
    params = {
        "api_key": api_key,
        "search": f"approvalID:{unii}",
        "limit": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None

    substance = data["results"][0]

    identifiers = {
        "unii": substance.get("approvalID"),
        "uuid": substance.get("uuid"),
        "preferred_name": None,
        "cas_numbers": [],
        "other_codes": {}
    }

    # 提取名稱
    if "names" in substance:
        for name in substance["names"]:
            if name.get("preferred"):
                identifiers["preferred_name"] = name.get("name")
                break
        if not identifiers["preferred_name"] and len(substance["names"]) > 0:
            identifiers["preferred_name"] = substance["names"][0].get("name")

    # 提取代碼
    if "codes" in substance:
        for code in substance["codes"]:
            code_system = code.get("codeSystem", "").upper()
            code_value = code.get("code")

            if "CAS" in code_system:
                identifiers["cas_numbers"].append(code_value)
            else:
                if code_system not in identifiers["other_codes"]:
                    identifiers["other_codes"][code_system] = []
                identifiers["other_codes"][code_system].append(code_value)

    return identifiers
```

### 化學結構查詢

```python
def get_chemical_structure(substance_name, api_key):
    """
    取得物質的化學結構資訊。

    參數：
        substance_name: 物質名稱
        api_key: FDA API 金鑰

    回傳：
        包含結構資訊的字典
    """
    import requests

    url = "https://api.fda.gov/other/substance.json"
    params = {
        "api_key": api_key,
        "search": f"names.name:{substance_name}",
        "limit": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None

    substance = data["results"][0]

    if "structure" not in substance:
        return None

    structure = substance["structure"]

    return {
        "smiles": structure.get("smiles"),
        "inchi": structure.get("inchi"),
        "inchi_key": structure.get("inchiKey"),
        "formula": structure.get("formula"),
        "molecular_weight": structure.get("molecularWeight"),
        "substance_class": substance.get("substanceClass")
    }
```

### 物質關係對應

```python
def get_substance_relationships(unii, api_key):
    """
    取得所有相關物質（代謝物、活性部分等）。

    參數：
        unii: FDA 唯一成分識別碼
        api_key: FDA API 金鑰

    回傳：
        按類型組織的關係字典
    """
    import requests

    url = "https://api.fda.gov/other/substance.json"
    params = {
        "api_key": api_key,
        "search": f"approvalID:{unii}",
        "limit": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None

    substance = data["results"][0]

    relationships = {}

    if "relationships" in substance:
        for rel in substance["relationships"]:
            rel_type = rel.get("type")
            if rel_type not in relationships:
                relationships[rel_type] = []

            related = {
                "uuid": rel.get("relatedSubstance", {}).get("uuid"),
                "unii": rel.get("relatedSubstance", {}).get("approvalID"),
                "name": rel.get("relatedSubstance", {}).get("refPname")
            }
            relationships[rel_type].append(related)

    return relationships
```

### 活性成分提取

```python
def find_active_ingredients_by_product(product_name, api_key):
    """
    查詢藥品中的活性成分。

    參數：
        product_name: 藥品名稱
        api_key: FDA API 金鑰

    回傳：
        活性成分 UNII 和名稱的列表
    """
    import requests

    # 首先搜尋藥物標籤資料庫
    label_url = "https://api.fda.gov/drug/label.json"
    label_params = {
        "api_key": api_key,
        "search": f"openfda.brand_name:{product_name}",
        "limit": 1
    }

    response = requests.get(label_url, params=label_params)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None

    label = data["results"][0]

    # 從 openfda 區段提取 UNII
    active_ingredients = []

    if "openfda" in label:
        openfda = label["openfda"]

        # 取得 UNII
        unii_list = openfda.get("unii", [])
        generic_names = openfda.get("generic_name", [])

        for i, unii in enumerate(unii_list):
            ingredient = {"unii": unii}
            if i < len(generic_names):
                ingredient["name"] = generic_names[i]

            # 取得額外的物質資訊
            substance_info = get_substance_identifiers(unii, api_key)
            if substance_info:
                ingredient.update(substance_info)

            active_ingredients.append(ingredient)

    return active_ingredients
```

## 最佳實踐

1. **使用 UNII 作為主要識別碼** - 跨 FDA 資料庫最一致
2. **在識別碼系統間對應** - CAS、UNII、InChI 鍵用於交叉參考
3. **處理物質變體** - 不同鹽型、水合物有不同的 UNII
4. **檢查物質類別** - 不同類別有不同的資料結構
5. **驗證化學結構** - 應驗證 SMILES 和 InChI
6. **考慮物質關係** - 活性部分與鹽型很重要
7. **使用首選名稱** - 比商品名更一致
8. **快取物質資料** - 物質資訊很少變更
9. **與其他端點交叉參考** - 連結物質與藥物/產品
10. **處理混合物成分** - 複雜產品有多個成分

## UNII 系統

FDA 唯一成分識別碼（UNII）系統提供：
- **唯一識別碼** - 每種物質獲得一個 UNII
- **物質特異性** - 不同形式（鹽、水合物）獲得不同的 UNII
- **全球認可** - 國際使用
- **穩定性** - UNII 一旦分配就不會改變
- **免費存取** - 無需授權

**UNII 格式**：10 字元字母數字代碼（例如：`R16CO5Y76E`）

## 物質類別說明

### Chemical（化學）
- 傳統小分子藥物
- 具有定義的分子結構
- 包括有機和無機化合物
- 可獲得 SMILES、InChI、分子式

### Protein（蛋白質）
- 多肽和蛋白質
- 可獲得序列資訊
- 可能有轉譯後修飾
- 包括抗體、酵素、激素

### Nucleic Acid（核酸）
- DNA 和 RNA 序列
- 寡核苷酸
- 反義、siRNA、mRNA
- 可獲得序列資料

### Polymer（聚合物）
- 合成和天然聚合物
- 結構重複單元
- 分子量分佈
- 用作賦形劑和活性成分

### Structurally Diverse（結構多樣性）
- 複雜的天然產物
- 植物提取物
- 沒有單一分子結構的材料
- 以來源和組成特徵描述

### Mixture（混合物）
- 已定義的物質組合
- 固定或可變組成
- 每個成分可追蹤

## 其他資源

- FDA 物質註冊系統：https://fdasis.nlm.nih.gov/srs/
- UNII 搜尋：https://precision.fda.gov/uniisearch
- OpenFDA 其他 API：https://open.fda.gov/apis/other/
- API 基礎：請參閱本參考目錄中的 `api_basics.md`
- Python 範例：請參閱 `scripts/fda_substance_query.py`
