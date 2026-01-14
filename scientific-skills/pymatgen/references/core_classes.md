# Pymatgen 核心類別參考

本參考記錄 `pymatgen.core` 中構成材料分析基礎的基本類別。

## 架構原則

Pymatgen 遵循物件導向設計，其中元素、位點和結構表示為物件。該框架強調晶體表示的週期性邊界條件，同時保持分子系統的彈性。

**單位慣例**：pymatgen 中的所有單位通常假設為原子單位：
- 長度：埃（Å）
- 能量：電子伏特（eV）
- 角度：度

## 元素和週期表

### Element
表示具有全面屬性的週期表元素。

**創建方法：**
```python
from pymatgen.core import Element

# 從符號創建
si = Element("Si")
# 從原子序創建
si = Element.from_Z(14)
# 從名稱創建
si = Element.from_name("silicon")
```

**關鍵屬性：**
- `atomic_mass`：原子質量（amu）
- `atomic_radius`：原子半徑（埃）
- `electronegativity`：Pauling 電負度
- `ionization_energy`：第一游離能（eV）
- `common_oxidation_states`：常見氧化態列表
- `is_metal`、`is_halogen`、`is_noble_gas` 等：布林屬性
- `X`：元素符號字串

### Species
擴展 Element 以用於帶電離子和特定氧化態。

```python
from pymatgen.core import Species

# 創建 Fe2+ 離子
fe2 = Species("Fe", 2)
# 或使用明確符號
fe2 = Species("Fe", +2)
```

### DummySpecies
用於特殊結構表示的佔位原子（例如空位）。

```python
from pymatgen.core import DummySpecies

vacancy = DummySpecies("X")
```

## Composition

表示化學式和組成，實現化學分析和操作。

### 創建
```python
from pymatgen.core import Composition

# 從字串公式
comp = Composition("Fe2O3")
# 從字典
comp = Composition({"Fe": 2, "O": 3})
# 從重量字典
comp = Composition.from_weight_dict({"Fe": 111.69, "O": 48.00})
```

### 關鍵方法
- `get_reduced_formula_and_factor()`：回傳簡化公式和乘法因子
- `oxi_state_guesses()`：嘗試確定氧化態
- `replace(replacements_dict)`：替換元素
- `add_charges_from_oxi_state_guesses()`：推斷並添加氧化態
- `is_element`：檢查組成是否為單一元素

### 關鍵屬性
- `weight`：分子量
- `reduced_formula`：簡化化學式
- `hill_formula`：Hill 表示法公式（C、H，然後按字母順序）
- `num_atoms`：原子總數
- `chemical_system`：按字母順序排列的元素（例如 "Fe-O"）
- `element_composition`：元素到數量的字典

## Lattice

定義晶體結構的晶胞幾何。

### 創建
```python
from pymatgen.core import Lattice

# 從晶格參數
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84,
                                  alpha=120, beta=90, gamma=60)

# 從矩陣（行向量為晶格向量）
lattice = Lattice([[3.84, 0, 0],
                   [0, 3.84, 0],
                   [0, 0, 3.84]])

# 立方晶格
lattice = Lattice.cubic(3.84)
# 六方晶格
lattice = Lattice.hexagonal(a=2.95, c=4.68)
```

### 關鍵方法
- `get_niggli_reduced_lattice()`：回傳 Niggli 簡化晶格
- `get_distance_and_image(frac_coords1, frac_coords2)`：具有週期性邊界條件的分數座標之間的距離
- `get_all_distances(frac_coords1, frac_coords2)`：包括週期性映像的距離

### 關鍵屬性
- `volume`：晶胞體積（Å³）
- `abc`：晶格參數（a, b, c）元組
- `angles`：晶格角度（alpha, beta, gamma）元組
- `matrix`：晶格向量的 3x3 矩陣
- `reciprocal_lattice`：倒晶格物件
- `is_orthogonal`：晶格向量是否正交

## Sites

### Site
表示非週期系統中的原子位置。

```python
from pymatgen.core import Site

site = Site("Si", [0.0, 0.0, 0.0])  # 物種和笛卡爾座標
```

### PeriodicSite
表示週期性晶格中具有分數座標的原子位置。

```python
from pymatgen.core import PeriodicSite

site = PeriodicSite("Si", [0.5, 0.5, 0.5], lattice)  # 物種、分數座標、晶格
```

**關鍵方法：**
- `distance(other_site)`：到另一個位點的距離
- `is_periodic_image(other_site)`：檢查位點是否為週期性映像

**關鍵屬性：**
- `species`：位點上的物種或元素
- `coords`：笛卡爾座標
- `frac_coords`：分數座標（用於 PeriodicSite）
- `x`、`y`、`z`：個別笛卡爾座標

## Structure

將晶體結構表示為週期性位點的集合。`Structure` 是可變的，而 `IStructure` 是不可變的。

### 創建
```python
from pymatgen.core import Structure, Lattice

# 從頭建立
coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84,
                                  alpha=120, beta=90, gamma=60)
struct = Structure(lattice, ["Si", "Si"], coords)

# 從檔案（自動格式偵測）
struct = Structure.from_file("POSCAR")
struct = Structure.from_file("structure.cif")

# 從空間群
struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.5),
                                   ["Si"], [[0, 0, 0]])
```

### 檔案 I/O
```python
# 寫入檔案（從副檔名推斷格式）
struct.to(filename="output.cif")
struct.to(filename="POSCAR")
struct.to(filename="structure.xyz")

# 取得字串表示
cif_string = struct.to(fmt="cif")
poscar_string = struct.to(fmt="poscar")
```

### 關鍵方法

**結構修改：**
- `append(species, coords)`：添加位點
- `insert(i, species, coords)`：在索引處插入位點
- `remove_sites(indices)`：按索引移除位點
- `replace(i, species)`：替換索引處的物種
- `apply_strain(strain)`：對結構施加應變
- `perturb(distance)`：隨機擾動原子位置
- `make_supercell(scaling_matrix)`：建立超晶胞
- `get_primitive_structure()`：取得原胞

**分析：**
- `get_distance(i, j)`：位點 i 和 j 之間的距離
- `get_neighbors(site, r)`：取得半徑 r 內的鄰居
- `get_all_neighbors(r)`：取得所有位點的所有鄰居
- `get_space_group_info()`：取得空間群資訊
- `matches(other_struct)`：檢查結構是否匹配

**內插：**
- `interpolate(end_structure, nimages)`：在結構之間內插

### 關鍵屬性
- `lattice`：Lattice 物件
- `species`：每個位點的物種列表
- `sites`：PeriodicSite 物件列表
- `num_sites`：位點數
- `volume`：結構體積
- `density`：密度（g/cm³）
- `composition`：Composition 物件
- `formula`：化學式
- `distance_matrix`：成對距離矩陣

## Molecule

表示非週期性原子集合。`Molecule` 是可變的，而 `IMolecule` 是不可變的。

### 創建
```python
from pymatgen.core import Molecule

# 從頭建立
coords = [[0.00, 0.00, 0.00],
          [0.00, 0.00, 1.08]]
mol = Molecule(["C", "O"], coords)

# 從檔案
mol = Molecule.from_file("molecule.xyz")
mol = Molecule.from_file("molecule.mol")
```

### 關鍵方法
- `get_covalent_bonds()`：根據共價半徑回傳鍵
- `get_neighbors(site, r)`：取得半徑內的鄰居
- `get_zmatrix()`：取得 Z-矩陣表示
- `get_distance(i, j)`：位點之間的距離
- `get_centered_molecule()`：將分子置於原點中心

### 關鍵屬性
- `species`：物種列表
- `sites`：Site 物件列表
- `num_sites`：原子數
- `charge`：分子總電荷
- `spin_multiplicity`：自旋多重態
- `center_of_mass`：質心座標

## 序列化

所有核心物件實現 `as_dict()` 和 `from_dict()` 方法，以實現穩健的 JSON/YAML 持久化。

```python
# 序列化為字典
struct_dict = struct.as_dict()

# 寫入 JSON
import json
with open("structure.json", "w") as f:
    json.dump(struct_dict, f)

# 從 JSON 讀取
with open("structure.json", "r") as f:
    struct_dict = json.load(f)
    struct = Structure.from_dict(struct_dict)
```

這種方法解決了 Python pickle 的限制，並保持跨 pymatgen 版本的相容性。

## 其他核心類別

### CovalentBond
表示分子中的鍵。

**關鍵屬性：**
- `length`：鍵長
- `get_bond_order()`：回傳鍵級（單鍵、雙鍵、三鍵）

### Ion
表示具有氧化態的帶電離子物種。

```python
from pymatgen.core import Ion

# 創建 Fe2+ 離子
fe2_ion = Ion.from_formula("Fe2+")
```

### Interface
表示用於異質結分析的基底-薄膜組合。

### GrainBoundary
表示晶界。

### Spectrum
表示光譜資料，具有標準化和處理方法。

**關鍵方法：**
- `normalize(mode="max")`：標準化光譜
- `smear(sigma)`：應用高斯展寬

## 最佳實務

1. **不可變性**：當結構不應修改時使用不可變版本（`IStructure`、`IMolecule`）
2. **序列化**：對於長期儲存偏好 `as_dict()`/`from_dict()` 而非 pickle
3. **單位**：始終使用原子單位（Å、eV）工作 - 轉換可在 `pymatgen.core.units` 中找到
4. **檔案 I/O**：使用 `from_file()` 進行自動格式偵測
5. **座標**：注意方法期望的是笛卡爾座標還是分數座標
