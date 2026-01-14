# Pymatgen 轉換和常見工作流程

本參考記錄 pymatgen 的轉換框架並提供常見材料科學工作流程的配方。

## 轉換框架

轉換提供一種系統化的方式來修改結構，同時追蹤修改歷史。

### 標準轉換

位於 `pymatgen.transformations.standard_transformations`。

#### SupercellTransformation

使用任意縮放矩陣建立超晶胞。

```python
from pymatgen.transformations.standard_transformations import SupercellTransformation

# 簡單 2x2x2 超晶胞
trans = SupercellTransformation([[2,0,0], [0,2,0], [0,0,2]])
new_struct = trans.apply_transformation(struct)

# 非正交超晶胞
trans = SupercellTransformation([[2,1,0], [0,2,0], [0,0,2]])
new_struct = trans.apply_transformation(struct)
```

#### SubstitutionTransformation

替換結構中的物種。

```python
from pymatgen.transformations.standard_transformations import SubstitutionTransformation

# 將所有 Fe 替換為 Mn
trans = SubstitutionTransformation({"Fe": "Mn"})
new_struct = trans.apply_transformation(struct)

# 部分取代（50% Fe -> Mn）
trans = SubstitutionTransformation({"Fe": {"Mn": 0.5, "Fe": 0.5}})
new_struct = trans.apply_transformation(struct)
```

#### RemoveSpeciesTransformation

從結構中移除特定物種。

```python
from pymatgen.transformations.standard_transformations import RemoveSpeciesTransformation

trans = RemoveSpeciesTransformation(["H"])  # 移除所有氫
new_struct = trans.apply_transformation(struct)
```

#### OrderDisorderedStructureTransformation

對具有部分佔據的無序結構進行有序化。

```python
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

trans = OrderDisorderedStructureTransformation()
new_struct = trans.apply_transformation(disordered_struct)
```

#### PrimitiveCellTransformation

轉換為原胞。

```python
from pymatgen.transformations.standard_transformations import PrimitiveCellTransformation

trans = PrimitiveCellTransformation()
primitive_struct = trans.apply_transformation(struct)
```

#### ConventionalCellTransformation

轉換為慣用晶胞。

```python
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

trans = ConventionalCellTransformation()
conventional_struct = trans.apply_transformation(struct)
```

#### RotationTransformation

旋轉結構。

```python
from pymatgen.transformations.standard_transformations import RotationTransformation

# 按軸和角度旋轉
trans = RotationTransformation([0, 0, 1], 45)  # 繞 z 軸旋轉 45°
new_struct = trans.apply_transformation(struct)
```

#### ScaleToRelaxedTransformation

縮放晶格以匹配弛豫結構。

```python
from pymatgen.transformations.standard_transformations import ScaleToRelaxedTransformation

trans = ScaleToRelaxedTransformation(relaxed_struct)
scaled_struct = trans.apply_transformation(unrelaxed_struct)
```

### 進階轉換

位於 `pymatgen.transformations.advanced_transformations`。

#### EnumerateStructureTransformation

從無序結構列舉所有對稱不等價的有序結構。

```python
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation

# 列舉最多每晶胞 8 個原子的結構
trans = EnumerateStructureTransformation(max_cell_size=8)
structures = trans.apply_transformation(struct, return_ranked_list=True)

# 回傳排序的結構列表
for s in structures[:5]:  # 前 5 個結構
    print(f"能量：{s['energy']}，結構：{s['structure']}")
```

#### MagOrderingTransformation

列舉磁性排序。

```python
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation

# 為每個物種指定磁矩
trans = MagOrderingTransformation({"Fe": 5.0, "Ni": 2.0})
mag_structures = trans.apply_transformation(struct, return_ranked_list=True)
```

#### DopingTransformation

系統化摻雜結構。

```python
from pymatgen.transformations.advanced_transformations import DopingTransformation

# 將 12.5% 的 Fe 位點替換為 Mn
trans = DopingTransformation("Mn", min_length=10)
doped_structs = trans.apply_transformation(struct, return_ranked_list=True)
```

#### ChargeBalanceTransformation

透過氧化態操作平衡結構中的電荷。

```python
from pymatgen.transformations.advanced_transformations import ChargeBalanceTransformation

trans = ChargeBalanceTransformation("Li")
charged_struct = trans.apply_transformation(struct)
```

#### SlabTransformation

生成表面切片。

```python
from pymatgen.transformations.advanced_transformations import SlabTransformation

trans = SlabTransformation(
    miller_index=[1, 0, 0],
    min_slab_size=10,
    min_vacuum_size=10,
    shift=0,
    lll_reduce=True
)
slab = trans.apply_transformation(struct)
```

### 鏈接轉換

```python
from pymatgen.alchemy.materials import TransformedStructure

# 建立追蹤歷史的轉換結構
ts = TransformedStructure(struct, [])

# 應用多個轉換
ts.append_transformation(SupercellTransformation([[2,0,0],[0,2,0],[0,0,2]]))
ts.append_transformation(SubstitutionTransformation({"Fe": "Mn"}))
ts.append_transformation(PrimitiveCellTransformation())

# 取得最終結構
final_struct = ts.final_structure

# 查看轉換歷史
print(ts.history)
```

## 常見工作流程

### 工作流程 1：高通量結構生成

為篩選研究生成多個結構。

```python
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import (
    SubstitutionTransformation,
    SupercellTransformation
)
from pymatgen.io.vasp.sets import MPRelaxSet

# 起始結構
base_struct = Structure.from_file("POSCAR")

# 定義取代
dopants = ["Mn", "Co", "Ni", "Cu"]
structures = {}

for dopant in dopants:
    # 建立取代結構
    trans = SubstitutionTransformation({"Fe": dopant})
    new_struct = trans.apply_transformation(base_struct)

    # 生成 VASP 輸入
    vasp_input = MPRelaxSet(new_struct)
    vasp_input.write_input(f"./calcs/Fe_{dopant}")

    structures[dopant] = new_struct

print(f"生成了 {len(structures)} 個結構")
```

### 工作流程 2：相圖建構

從 Materials Project 資料建構和分析相圖。

```python
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.core import Composition

# 從 Materials Project 取得資料
with MPRester() as mpr:
    entries = mpr.get_entries_in_chemsys("Li-Fe-O")

# 建構相圖
pd = PhaseDiagram(entries)

# 分析特定組成
comp = Composition("LiFeO2")
e_above_hull = pd.get_e_above_hull(entries[0])

# 取得分解產物
decomp = pd.get_decomposition(comp)
print(f"分解：{decomp}")

# 視覺化
plotter = PDPlotter(pd)
plotter.show()
```

### 工作流程 3：表面能計算

從切片計算計算表面能。

```python
from pymatgen.core.surface import SlabGenerator, generate_all_slabs
from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
from pymatgen.core import Structure

# 讀取塊材結構
bulk = Structure.from_file("bulk_POSCAR")

# 取得塊材能量（從先前計算）
from pymatgen.io.vasp import Vasprun
bulk_vasprun = Vasprun("bulk/vasprun.xml")
bulk_energy_per_atom = bulk_vasprun.final_energy / len(bulk)

# 生成切片
miller_indices = [(1,0,0), (1,1,0), (1,1,1)]
surface_energies = {}

for miller in miller_indices:
    slabgen = SlabGenerator(
        bulk,
        miller_index=miller,
        min_slab_size=10,
        min_vacuum_size=15,
        center_slab=True
    )

    slab = slabgen.get_slabs()[0]

    # 為切片寫入 VASP 輸入
    relax = MPRelaxSet(slab)
    relax.write_input(f"./slab_{miller[0]}{miller[1]}{miller[2]}")

    # 計算後，計算表面能：
    # slab_vasprun = Vasprun(f"slab_{miller[0]}{miller[1]}{miller[2]}/vasprun.xml")
    # slab_energy = slab_vasprun.final_energy
    # n_atoms = len(slab)
    # area = slab.surface_area  # 以 Å² 為單位
    #
    # # 表面能（J/m²）
    # surf_energy = (slab_energy - n_atoms * bulk_energy_per_atom) / (2 * area)
    # surf_energy *= 16.021766  # 將 eV/Å² 轉換為 J/m²
    # surface_energies[miller] = surf_energy

print(f"為 {len(miller_indices)} 個表面設置了計算")
```

### 工作流程 4：能帶結構計算

能帶結構計算的完整工作流程。

```python
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet
from pymatgen.symmetry.bandstructure import HighSymmKpath

# 步驟 1：弛豫
struct = Structure.from_file("initial_POSCAR")
relax = MPRelaxSet(struct)
relax.write_input("./1_relax")

# 弛豫後，讀取結構
relaxed_struct = Structure.from_file("1_relax/CONTCAR")

# 步驟 2：靜態計算
static = MPStaticSet(relaxed_struct)
static.write_input("./2_static")

# 步驟 3：能帶結構（非自洽）
kpath = HighSymmKpath(relaxed_struct)
nscf = MPNonSCFSet(relaxed_struct, mode="line")  # 能帶結構模式
nscf.write_input("./3_bandstructure")

# 計算後，分析
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter

vasprun = Vasprun("3_bandstructure/vasprun.xml")
bs = vasprun.get_band_structure(line_mode=True)

print(f"能隙：{bs.get_band_gap()}")

plotter = BSPlotter(bs)
plotter.save_plot("band_structure.png")
```

### 工作流程 5：分子動力學設置

設置和分析分子動力學模擬。

```python
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MVLRelaxSet
from pymatgen.io.vasp.inputs import Incar

# 讀取結構
struct = Structure.from_file("POSCAR")

# 為 MD 建立 2x2x2 超晶胞
from pymatgen.transformations.standard_transformations import SupercellTransformation
trans = SupercellTransformation([[2,0,0],[0,2,0],[0,0,2]])
supercell = trans.apply_transformation(struct)

# 設置 VASP 輸入
md_input = MVLRelaxSet(supercell)

# 為 MD 修改 INCAR
incar = md_input.incar
incar.update({
    "IBRION": 0,      # 分子動力學
    "NSW": 1000,      # 步數
    "POTIM": 2,       # 時間步長（fs）
    "TEBEG": 300,     # 初始溫度（K）
    "TEEND": 300,     # 最終溫度（K）
    "SMASS": 0,       # NVT 系綜
    "MDALGO": 2,      # Nose-Hoover 恆溫器
})

md_input.incar = incar
md_input.write_input("./md_calc")
```

### 工作流程 6：擴散分析

從 AIMD 軌跡分析離子擴散。

```python
from pymatgen.io.vasp import Xdatcar
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

# 從 XDATCAR 讀取軌跡
xdatcar = Xdatcar("XDATCAR")
structures = xdatcar.structures

# 分析特定物種（例如 Li）的擴散
analyzer = DiffusionAnalyzer.from_structures(
    structures,
    specie="Li",
    temperature=300,  # K
    time_step=2,      # fs
    step_skip=10      # 跳過初始平衡
)

# 取得擴散係數
diffusivity = analyzer.diffusivity  # cm²/s
conductivity = analyzer.conductivity  # mS/cm

# 取得均方位移
msd = analyzer.msd

# 繪製 MSD
analyzer.plot_msd()

print(f"擴散係數：{diffusivity:.2e} cm²/s")
print(f"電導率：{conductivity:.2e} mS/cm")
```

### 工作流程 7：結構預測和列舉

預測和列舉可能的結構。

```python
from pymatgen.core import Structure, Lattice
from pymatgen.transformations.advanced_transformations import (
    EnumerateStructureTransformation,
    SubstitutionTransformation
)

# 從已知結構類型開始（例如岩鹽）
lattice = Lattice.cubic(4.2)
struct = Structure.from_spacegroup("Fm-3m", lattice, ["Li", "O"], [[0,0,0], [0.5,0.5,0.5]])

# 建立無序結構
from pymatgen.core import Species
species_on_site = {Species("Li"): 0.5, Species("Na"): 0.5}
struct[0] = species_on_site  # Li 位點上的混合佔據

# 列舉所有有序結構
trans = EnumerateStructureTransformation(max_cell_size=4)
ordered_structs = trans.apply_transformation(struct, return_ranked_list=True)

print(f"找到 {len(ordered_structs)} 個不同的有序結構")

# 寫入所有結構
for i, s_dict in enumerate(ordered_structs[:10]):  # 前 10 個
    s_dict['structure'].to(filename=f"ordered_struct_{i}.cif")
```

### 工作流程 8：彈性常數計算

使用應力-應變方法計算彈性常數。

```python
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.io.vasp.sets import MPStaticSet

# 讀取平衡結構
struct = Structure.from_file("relaxed_POSCAR")

# 生成變形結構
strains = [0.00, 0.01, 0.02, -0.01, -0.02]  # 施加的應變
deformation_sets = []

for strain in strains:
    # 在不同方向施加應變
    trans = DeformStructureTransformation([[1+strain, 0, 0], [0, 1, 0], [0, 0, 1]])
    deformed = trans.apply_transformation(struct)

    # 設置 VASP 計算
    static = MPStaticSet(deformed)
    static.write_input(f"./strain_{strain:.2f}")

# 計算後，擬合應力與應變以取得彈性常數
# from pymatgen.analysis.elasticity import ElasticTensor
# ...（從 OUTCAR 收集應力張量）
# elastic_tensor = ElasticTensor.from_stress_list(stress_list)
```

### 工作流程 9：吸附能計算

計算表面上的吸附能。

```python
from pymatgen.core import Structure, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.vasp.sets import MPRelaxSet

# 生成切片
bulk = Structure.from_file("bulk_POSCAR")
slabgen = SlabGenerator(bulk, (1,1,1), 10, 10)
slab = slabgen.get_slabs()[0]

# 尋找吸附位點
asf = AdsorbateSiteFinder(slab)
ads_sites = asf.find_adsorption_sites()

# 建立吸附物
adsorbate = Molecule("O", [[0, 0, 0]])

# 生成帶有吸附物的結構
ads_structs = asf.add_adsorbate(adsorbate, ads_sites["ontop"][0])

# 設置計算
relax_slab = MPRelaxSet(slab)
relax_slab.write_input("./slab")

relax_ads = MPRelaxSet(ads_structs)
relax_ads.write_input("./slab_with_adsorbate")

# 計算後：
# E_ads = E(切片+吸附物) - E(切片) - E(氣態吸附物)
```

### 工作流程 10：高通量材料篩選

篩選材料資料庫以尋找特定性質。

```python
from mp_api.client import MPRester
from pymatgen.core import Structure
import pandas as pd

# 定義篩選標準
def screen_material(material):
    """篩選潛在的電池正極材料"""
    criteria = {
        "has_li": "Li" in material.composition.elements,
        "stable": material.energy_above_hull < 0.05,
        "good_voltage": 2.5 < material.formation_energy_per_atom < 4.5,
        "electronically_conductive": material.band_gap < 0.5
    }
    return all(criteria.values()), criteria

# 查詢 Materials Project
with MPRester() as mpr:
    # 取得潛在材料
    materials = mpr.materials.summary.search(
        elements=["Li"],
        energy_above_hull=(0, 0.05),
    )

    results = []
    for mat in materials:
        passes, criteria = screen_material(mat)
        if passes:
            results.append({
                "material_id": mat.material_id,
                "formula": mat.formula_pretty,
                "energy_above_hull": mat.energy_above_hull,
                "band_gap": mat.band_gap,
            })

    # 儲存結果
    df = pd.DataFrame(results)
    df.to_csv("screened_materials.csv", index=False)

    print(f"找到 {len(results)} 個有希望的材料")
```

## 工作流程最佳實務

1. **模組化設計**：將工作流程分解為離散步驟
2. **錯誤處理**：檢查檔案存在和計算收斂
3. **文件**：使用 `TransformedStructure` 追蹤轉換歷史
4. **版本控制**：將輸入參數和腳本儲存在 git 中
5. **自動化**：使用工作流程管理器（Fireworks、AiiDA）進行複雜管線
6. **資料管理**：以清晰的目錄結構組織計算
7. **驗證**：在繼續之前始終驗證中間結果

## 與工作流程工具整合

Pymatgen 與幾個工作流程管理系統整合：

- **Atomate**：預建的 VASP 工作流程
- **Fireworks**：工作流程執行引擎
- **AiiDA**：來源追蹤和工作流程管理
- **Custodian**：錯誤更正和作業監控

這些工具為正式計算提供穩健的自動化。
