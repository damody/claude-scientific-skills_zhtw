# Materials Project API 參考

本參考記錄如何透過 pymatgen 的 API 整合存取和使用 Materials Project 資料庫。

## 概述

Materials Project 是一個全面的計算材料性質資料庫，包含數十萬種無機晶體和分子的資料。API 透過 `MPRester` 客戶端提供程式化存取這些資料。

## 安裝和設置

Materials Project API 客戶端現在在獨立套件中：

```bash
pip install mp-api
```

### 取得 API 金鑰

1. 訪問 https://next-gen.materialsproject.org/
2. 創建帳戶或登入
3. 導航到您的儀表板/設定
4. 生成 API 金鑰
5. 將其儲存為環境變數：

```bash
export MP_API_KEY="your_api_key_here"
```

或添加到您的 shell 配置檔案（~/.bashrc、~/.zshrc 等）

## 基本用法

### 初始化

```python
from mp_api.client import MPRester

# 使用環境變數（推薦）
with MPRester() as mpr:
    # 執行查詢
    pass

# 或明確傳遞 API 金鑰
with MPRester("your_api_key_here") as mpr:
    # 執行查詢
    pass
```

**重要**：始終使用 `with` 上下文管理器以確保會話正確關閉。

## 查詢材料資料

### 按化學式搜尋

```python
with MPRester() as mpr:
    # 取得具有該化學式的所有材料
    materials = mpr.materials.summary.search(formula="Fe2O3")

    for mat in materials:
        print(f"材料 ID：{mat.material_id}")
        print(f"化學式：{mat.formula_pretty}")
        print(f"高於凸包的能量：{mat.energy_above_hull} eV/atom")
        print(f"能隙：{mat.band_gap} eV")
        print()
```

### 按材料 ID 搜尋

```python
with MPRester() as mpr:
    # 取得特定材料
    material = mpr.materials.summary.search(material_ids=["mp-149"])[0]

    print(f"化學式：{material.formula_pretty}")
    print(f"空間群：{material.symmetry.symbol}")
    print(f"密度：{material.density} g/cm³")
```

### 按化學系統搜尋

```python
with MPRester() as mpr:
    # 取得 Fe-O 系統中的所有材料
    materials = mpr.materials.summary.search(chemsys="Fe-O")

    # 取得三元系統中的材料
    materials = mpr.materials.summary.search(chemsys="Li-Fe-O")
```

### 按元素搜尋

```python
with MPRester() as mpr:
    # 包含 Fe 和 O 的材料
    materials = mpr.materials.summary.search(elements=["Fe", "O"])

    # 僅包含 Fe 和 O 的材料（排除其他）
    materials = mpr.materials.summary.search(
        elements=["Fe", "O"],
        exclude_elements=True
    )
```

## 取得結構

### 從材料 ID 取得結構

```python
with MPRester() as mpr:
    # 取得結構
    structure = mpr.get_structure_by_material_id("mp-149")

    # 取得多個結構
    structures = mpr.get_structures(["mp-149", "mp-510", "mp-19017"])
```

### 某化學式的所有結構

```python
with MPRester() as mpr:
    # 取得所有 Fe2O3 結構
    materials = mpr.materials.summary.search(formula="Fe2O3")

    for mat in materials:
        structure = mpr.get_structure_by_material_id(mat.material_id)
        print(f"{mat.material_id}：{structure.get_space_group_info()}")
```

## 進階查詢

### 屬性過濾

```python
with MPRester() as mpr:
    # 具有特定屬性範圍的材料
    materials = mpr.materials.summary.search(
        chemsys="Li-Fe-O",
        energy_above_hull=(0, 0.05),  # 穩定或近穩定
        band_gap=(1.0, 3.0),           # 半導體
    )

    # 磁性材料
    materials = mpr.materials.summary.search(
        elements=["Fe"],
        is_magnetic=True
    )

    # 僅金屬
    materials = mpr.materials.summary.search(
        chemsys="Fe-Ni",
        is_metal=True
    )
```

### 排序和限制

```python
with MPRester() as mpr:
    # 取得最穩定的材料
    materials = mpr.materials.summary.search(
        chemsys="Li-Fe-O",
        sort_fields=["energy_above_hull"],
        num_chunks=1,
        chunk_size=10  # 限制為 10 個結果
    )
```

## 電子結構資料

### 能帶結構

```python
with MPRester() as mpr:
    # 取得能帶結構
    bs = mpr.get_bandstructure_by_material_id("mp-149")

    # 分析能帶結構
    if bs:
        print(f"能隙：{bs.get_band_gap()}")
        print(f"是否為金屬：{bs.is_metal()}")
        print(f"直接能隙：{bs.get_band_gap()['direct']}")

        # 繪圖
        from pymatgen.electronic_structure.plotter import BSPlotter
        plotter = BSPlotter(bs)
        plotter.show()
```

### 態密度

```python
with MPRester() as mpr:
    # 取得 DOS
    dos = mpr.get_dos_by_material_id("mp-149")

    if dos:
        # 從 DOS 取得能隙
        gap = dos.get_gap()
        print(f"從 DOS 得到的能隙：{gap} eV")

        # 繪製 DOS
        from pymatgen.electronic_structure.plotter import DosPlotter
        plotter = DosPlotter()
        plotter.add_dos("Total DOS", dos)
        plotter.show()
```

### 費米面

```python
with MPRester() as mpr:
    # 取得費米面的電子結構資料
    bs = mpr.get_bandstructure_by_material_id("mp-149", line_mode=False)
```

## 熱力學資料

### 相圖建構

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

with MPRester() as mpr:
    # 取得相圖條目
    entries = mpr.get_entries_in_chemsys("Li-Fe-O")

    # 建構相圖
    pd = PhaseDiagram(entries)

    # 繪圖
    plotter = PDPlotter(pd)
    plotter.show()
```

### Pourbaix 圖

```python
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter

with MPRester() as mpr:
    # 取得 Pourbaix 圖條目
    entries = mpr.get_pourbaix_entries(["Fe"])

    # 建構 Pourbaix 圖
    pb = PourbaixDiagram(entries)

    # 繪圖
    plotter = PourbaixPlotter(pb)
    plotter.show()
```

### 形成能

```python
with MPRester() as mpr:
    materials = mpr.materials.summary.search(material_ids=["mp-149"])

    for mat in materials:
        print(f"形成能：{mat.formation_energy_per_atom} eV/atom")
        print(f"高於凸包的能量：{mat.energy_above_hull} eV/atom")
```

## 彈性和機械性質

```python
with MPRester() as mpr:
    # 搜尋具有彈性資料的材料
    materials = mpr.materials.elasticity.search(
        chemsys="Fe-O",
        bulk_modulus_vrh=(100, 300)  # GPa
    )

    for mat in materials:
        print(f"{mat.material_id}：K = {mat.bulk_modulus_vrh} GPa")
```

## 介電性質

```python
with MPRester() as mpr:
    # 取得介電資料
    materials = mpr.materials.dielectric.search(
        material_ids=["mp-149"]
    )

    for mat in materials:
        print(f"介電常數：{mat.e_electronic}")
        print(f"折射率：{mat.n}")
```

## 壓電性質

```python
with MPRester() as mpr:
    # 取得壓電材料
    materials = mpr.materials.piezoelectric.search(
        piezoelectric_modulus=(1, 100)
    )
```

## 表面性質

```python
with MPRester() as mpr:
    # 取得表面資料
    surfaces = mpr.materials.surface_properties.search(
        material_ids=["mp-149"]
    )
```

## 分子資料（用於分子材料）

```python
with MPRester() as mpr:
    # 搜尋分子
    molecules = mpr.molecules.summary.search(
        formula="H2O"
    )

    for mol in molecules:
        print(f"分子 ID：{mol.molecule_id}")
        print(f"化學式：{mol.formula_pretty}")
```

## 批次資料下載

### 下載材料的所有資料

```python
with MPRester() as mpr:
    # 取得全面資料
    materials = mpr.materials.summary.search(
        material_ids=["mp-149"],
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "energy_above_hull",
            "band_gap",
            "density",
            "symmetry",
            "elasticity",
            "magnetic_ordering"
        ]
    )
```

## 來源和計算詳情

```python
with MPRester() as mpr:
    # 取得計算詳情
    materials = mpr.materials.summary.search(
        material_ids=["mp-149"],
        fields=["material_id", "origins"]
    )

    for mat in materials:
        print(f"來源：{mat.origins}")
```

## 處理條目

### 用於熱力學分析的 ComputedEntry

```python
with MPRester() as mpr:
    # 取得條目（包括能量和組成）
    entries = mpr.get_entries_in_chemsys("Li-Fe-O")

    # 條目可直接用於相圖分析
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    pd = PhaseDiagram(entries)

    # 檢查穩定性
    for entry in entries[:5]:
        e_above_hull = pd.get_e_above_hull(entry)
        print(f"{entry.composition.reduced_formula}：{e_above_hull:.3f} eV/atom")
```

## 速率限制和最佳實務

### 速率限制

Materials Project API 有速率限制以確保公平使用：
- 注意請求頻率
- 盡可能使用批次查詢
- 將結果快取到本地以進行重複分析

### 高效查詢

```python
# 差：多個單獨查詢
with MPRester() as mpr:
    for mp_id in ["mp-149", "mp-510", "mp-19017"]:
        struct = mpr.get_structure_by_material_id(mp_id)  # 3 次 API 呼叫

# 好：單次批次查詢
with MPRester() as mpr:
    structs = mpr.get_structures(["mp-149", "mp-510", "mp-19017"])  # 1 次 API 呼叫
```

### 快取結果

```python
import json

# 儲存結果供以後使用
with MPRester() as mpr:
    materials = mpr.materials.summary.search(chemsys="Li-Fe-O")

    # 儲存到檔案
    with open("li_fe_o_materials.json", "w") as f:
        json.dump([mat.dict() for mat in materials], f)

# 載入快取結果
with open("li_fe_o_materials.json", "r") as f:
    cached_data = json.load(f)
```

## 錯誤處理

```python
from mp_api.client.core.client import MPRestError

try:
    with MPRester() as mpr:
        materials = mpr.materials.summary.search(material_ids=["invalid-id"])
except MPRestError as e:
    print(f"API 錯誤：{e}")
except Exception as e:
    print(f"意外錯誤：{e}")
```

## 常見使用情境

### 尋找穩定化合物

```python
with MPRester() as mpr:
    # 取得化學系統中所有穩定化合物
    materials = mpr.materials.summary.search(
        chemsys="Li-Fe-O",
        energy_above_hull=(0, 0.001)  # 基本上在凸包上
    )

    print(f"找到 {len(materials)} 個穩定化合物")
    for mat in materials:
        print(f"  {mat.formula_pretty}（{mat.material_id}）")
```

### 電池材料篩選

```python
with MPRester() as mpr:
    # 篩選潛在的正極材料
    materials = mpr.materials.summary.search(
        elements=["Li"],  # 必須包含 Li
        energy_above_hull=(0, 0.05),  # 近穩定
        band_gap=(0, 0.5),  # 金屬或小能隙
    )

    print(f"找到 {len(materials)} 個潛在正極材料")
```

### 尋找具有特定晶體結構的材料

```python
with MPRester() as mpr:
    # 尋找具有特定空間群的材料
    materials = mpr.materials.summary.search(
        chemsys="Fe-O",
        spacegroup_number=167  # R-3c（剛玉結構）
    )
```

## 與其他 Pymatgen 功能整合

從 Materials Project 檢索的所有資料都可以直接用於 pymatgen 的分析工具：

```python
with MPRester() as mpr:
    # 取得結構
    struct = mpr.get_structure_by_material_id("mp-149")

    # 使用 pymatgen 分析
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(struct)

    # 生成表面
    from pymatgen.core.surface import SlabGenerator
    slabgen = SlabGenerator(struct, (1,0,0), 10, 10)
    slabs = slabgen.get_slabs()

    # 相圖分析
    entries = mpr.get_entries_in_chemsys(struct.composition.chemical_system)
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    pd = PhaseDiagram(entries)
```

## 其他資源

- **API 文件**：https://docs.materialsproject.org/
- **Materials Project 網站**：https://next-gen.materialsproject.org/
- **GitHub**：https://github.com/materialsproject/api
- **論壇**：https://matsci.org/

## 最佳實務摘要

1. **始終使用上下文管理器**：使用 `with MPRester() as mpr:`
2. **將 API 金鑰儲存為環境變數**：切勿硬編碼 API 金鑰
3. **批次查詢**：盡可能一次請求多個項目
4. **快取結果**：將常用資料儲存在本地
5. **處理錯誤**：將 API 呼叫包裝在 try-except 區塊中
6. **具體化**：使用過濾器限制結果並減少資料傳輸
7. **檢查資料可用性**：並非所有材料都有所有屬性
