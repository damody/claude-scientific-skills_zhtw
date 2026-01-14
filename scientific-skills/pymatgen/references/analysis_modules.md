# Pymatgen 分析模組參考

本參考記錄 pymatgen 廣泛的材料表徵、性質預測和計算分析能力。

## 相圖和熱力學

### 相圖建構

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.computed_entries import ComputedEntry

# 建立條目（組成和每原子能量）
entries = [
    ComputedEntry("Fe", -8.4),
    ComputedEntry("O2", -4.9),
    ComputedEntry("FeO", -6.7),
    ComputedEntry("Fe2O3", -8.3),
    ComputedEntry("Fe3O4", -9.1),
]

# 建構相圖
pd = PhaseDiagram(entries)

# 取得穩定條目
stable_entries = pd.stable_entries

# 取得高於凸包的能量（穩定性）
entry_to_test = ComputedEntry("Fe2O3", -8.0)
energy_above_hull = pd.get_e_above_hull(entry_to_test)

# 取得分解產物
decomp = pd.get_decomposition(entry_to_test.composition)
# 回傳：{entry1: fraction1, entry2: fraction2, ...}

# 取得平衡反應能量
rxn_energy = pd.get_equilibrium_reaction_energy(entry_to_test)

# 繪製相圖
plotter = PDPlotter(pd)
plotter.show()
plotter.write_image("phase_diagram.png")
```

### 化學勢圖

```python
from pymatgen.analysis.phase_diagram import ChemicalPotentialDiagram

# 建立化學勢圖
cpd = ChemicalPotentialDiagram(entries, limits={"O": (-10, 0)})

# 取得區域（穩定區域）
domains = cpd.domains
```

### Pourbaix 圖

具有 pH 和電位軸的電化學相圖。

```python
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter
from pymatgen.entries.computed_entries import ComputedEntry

# 建立包含水溶液物種修正的條目
entries = [...]  # 包括固體和離子

# 建構 Pourbaix 圖
pb = PourbaixDiagram(entries)

# 取得特定 pH 和電位下的穩定條目
stable_entry = pb.get_stable_entry(pH=7, V=0)

# 繪圖
plotter = PourbaixPlotter(pb)
plotter.show()
```

## 結構分析

### 結構匹配和比較

```python
from pymatgen.analysis.structure_matcher import StructureMatcher

matcher = StructureMatcher()

# 檢查結構是否匹配
is_match = matcher.fit(struct1, struct2)

# 取得結構之間的對應
mapping = matcher.get_mapping(struct1, struct2)

# 分組相似結構
grouped = matcher.group_structures([struct1, struct2, struct3, ...])
```

### Ewald 求和

計算離子結構的靜電能。

```python
from pymatgen.analysis.ewald import EwaldSummation

ewald = EwaldSummation(struct)
total_energy = ewald.total_energy  # 以 eV 為單位
forces = ewald.forces  # 每個位點上的力
```

### 對稱性分析

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sga = SpacegroupAnalyzer(struct)

# 取得空間群資訊
spacegroup_symbol = sga.get_space_group_symbol()  # 例如 "Fm-3m"
spacegroup_number = sga.get_space_group_number()   # 例如 225
crystal_system = sga.get_crystal_system()           # 例如 "cubic"

# 取得對稱化結構
sym_struct = sga.get_symmetrized_structure()
equivalent_sites = sym_struct.equivalent_sites

# 取得慣用/原胞
conventional = sga.get_conventional_standard_structure()
primitive = sga.get_primitive_standard_structure()

# 取得對稱操作
symmetry_ops = sga.get_symmetry_operations()
```

## 局部環境分析

### 配位環境

```python
from pymatgen.analysis.local_env import (
    VoronoiNN,           # Voronoi 鑲嵌
    CrystalNN,           # 基於晶體
    MinimumDistanceNN,   # 距離截斷
    BrunnerNN_real,      # Brunner 方法
)

# Voronoi 最近鄰
voronoi = VoronoiNN()
neighbors = voronoi.get_nn_info(struct, n=0)  # 位置 0 的鄰居

# CrystalNN（大多數情況推薦）
crystalnn = CrystalNN()
neighbors = crystalnn.get_nn_info(struct, n=0)

# 分析所有位點
for i, site in enumerate(struct):
    neighbors = voronoi.get_nn_info(struct, i)
    coordination_number = len(neighbors)
    print(f"位點 {i}（{site.species_string}）：配位數 = {coordination_number}")
```

### 配位幾何（ChemEnv）

詳細的配位環境識別。

```python
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy

lgf = LocalGeometryFinder()
lgf.setup_structure(struct)

# 取得位點的配位環境
se = lgf.compute_structure_environments(only_indices=[0])
strategy = SimplestChemenvStrategy()
lse = strategy.get_site_coordination_environment(se[0])

print(f"配位：{lse}")
```

### 鍵價和

```python
from pymatgen.analysis.bond_valence import BVAnalyzer

bva = BVAnalyzer()

# 計算氧化態
valences = bva.get_valences(struct)

# 取得帶有氧化態的結構
struct_with_oxi = bva.get_oxi_state_decorated_structure(struct)
```

## 表面和介面分析

### 表面（切片）生成

```python
from pymatgen.core.surface import SlabGenerator, generate_all_slabs

# 為特定 Miller 指數生成切片
slabgen = SlabGenerator(
    struct,
    miller_index=(1, 1, 1),
    min_slab_size=10.0,     # 最小切片厚度（Å）
    min_vacuum_size=10.0,   # 最小真空厚度（Å）
    center_slab=True
)

slabs = slabgen.get_slabs()

# 生成直到某 Miller 指數的所有切片
all_slabs = generate_all_slabs(
    struct,
    max_index=2,
    min_slab_size=10.0,
    min_vacuum_size=10.0
)
```

### Wulff 形狀建構

```python
from pymatgen.analysis.wulff import WulffShape

# 定義表面能（J/m²）
surface_energies = {
    (1, 0, 0): 1.0,
    (1, 1, 0): 1.1,
    (1, 1, 1): 0.9,
}

wulff = WulffShape(struct.lattice, surface_energies, symm_reduce=True)

# 取得有效半徑和表面積
effective_radius = wulff.effective_radius
surface_area = wulff.surface_area
volume = wulff.volume

# 視覺化
wulff.show()
```

### 吸附位點尋找

```python
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

asf = AdsorbateSiteFinder(slab)

# 尋找吸附位點
ads_sites = asf.find_adsorption_sites()
# 回傳字典：{"ontop": [...], "bridge": [...], "hollow": [...]}

# 生成帶有吸附物的結構
from pymatgen.core import Molecule
adsorbate = Molecule("O", [[0, 0, 0]])

ads_structs = asf.generate_adsorption_structures(
    adsorbate,
    repeat=[2, 2, 1],  # 超晶胞以減少吸附物覆蓋率
)
```

### 介面建構

```python
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

# 建構兩種材料之間的介面
builder = CoherentInterfaceBuilder(
    substrate_structure=substrate,
    film_structure=film,
    substrate_miller=(0, 0, 1),
    film_miller=(1, 1, 1),
)

interfaces = builder.get_interfaces()
```

## 磁性

### 磁性結構分析

```python
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer

analyzer = CollinearMagneticStructureAnalyzer(struct)

# 取得磁性排序
ordering = analyzer.ordering  # 例如 "FM"（鐵磁）、"AFM"、"FiM"

# 取得磁性空間群
mag_space_group = analyzer.get_structure_with_spin().get_space_group_info()
```

### 磁性排序列舉

```python
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation

# 列舉可能的磁性排序
mag_trans = MagOrderingTransformation({"Fe": 5.0})  # 磁矩以 μB 為單位
transformed_structures = mag_trans.apply_transformation(struct, return_ranked_list=True)
```

## 電子結構分析

### 能帶結構分析

```python
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.plotter import BSPlotter

# 從 VASP 計算讀取能帶結構
from pymatgen.io.vasp import Vasprun
vasprun = Vasprun("vasprun.xml")
bs = vasprun.get_band_structure()

# 取得能隙
band_gap = bs.get_band_gap()
# 回傳：{'energy': gap_value, 'direct': True/False, 'transition': '...'}

# 檢查是否為金屬
is_metal = bs.is_metal()

# 取得價帶頂和導帶底
vbm = bs.get_vbm()
cbm = bs.get_cbm()

# 繪製能帶結構
plotter = BSPlotter(bs)
plotter.show()
plotter.save_plot("band_structure.png")
```

### 態密度（DOS）

```python
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.plotter import DosPlotter

# 從 VASP 計算讀取 DOS
vasprun = Vasprun("vasprun.xml")
dos = vasprun.complete_dos

# 取得總 DOS
total_dos = dos.densities

# 取得投影 DOS
pdos = dos.get_element_dos()  # 按元素
site_dos = dos.get_site_dos(struct[0])  # 特定位點
spd_dos = dos.get_spd_dos()  # 按軌域（s、p、d）

# 繪製 DOS
plotter = DosPlotter()
plotter.add_dos("Total", dos)
plotter.show()
```

### 費米面

```python
from pymatgen.electronic_structure.boltztrap2 import BoltztrapRunner

runner = BoltztrapRunner(struct, nelec=n_electrons)
runner.run()

# 取得不同溫度下的輸運性質
results = runner.get_results()
```

## 繞射

### X 射線繞射（XRD）

```python
from pymatgen.analysis.diffraction.xrd import XRDCalculator

xrd = XRDCalculator()

pattern = xrd.get_pattern(struct, two_theta_range=(0, 90))

# 取得峰資料
for peak in pattern.hkls:
    print(f"2θ = {peak['2theta']:.2f}°, hkl = {peak['hkl']}, I = {peak['intensity']:.1f}")

# 繪製圖譜
pattern.plot()
```

### 中子繞射

```python
from pymatgen.analysis.diffraction.neutron import NDCalculator

nd = NDCalculator()
pattern = nd.get_pattern(struct)
```

## 彈性和機械性質

```python
from pymatgen.analysis.elasticity import ElasticTensor, Stress, Strain

# 從矩陣建立彈性張量
elastic_tensor = ElasticTensor([[...]])  # 6x6 或 3x3x3x3 矩陣

# 取得機械性質
bulk_modulus = elastic_tensor.k_voigt  # Voigt 體積模量（GPa）
shear_modulus = elastic_tensor.g_voigt  # 剪切模量（GPa）
youngs_modulus = elastic_tensor.y_mod  # 楊氏模量（GPa）

# 施加應變
strain = Strain([[0.01, 0, 0], [0, 0, 0], [0, 0, 0]])
stress = elastic_tensor.calculate_stress(strain)
```

## 反應分析

### 反應計算

```python
from pymatgen.analysis.reaction_calculator import ComputedReaction

reactants = [ComputedEntry("Fe", -8.4), ComputedEntry("O2", -4.9)]
products = [ComputedEntry("Fe2O3", -8.3)]

rxn = ComputedReaction(reactants, products)

# 取得平衡方程式
balanced_rxn = rxn.normalized_repr  # 例如 "2 Fe + 1.5 O2 -> Fe2O3"

# 取得反應能量
energy = rxn.calculated_reaction_energy  # 每公式單位 eV
```

### 反應路徑尋找

```python
from pymatgen.analysis.path_finder import ChgcarPotential, NEBPathfinder

# 讀取電荷密度
chgcar_potential = ChgcarPotential.from_file("CHGCAR")

# 尋找擴散路徑
neb_path = NEBPathfinder(
    start_struct,
    end_struct,
    relax_sites=[i for i in range(len(start_struct))],
    v=chgcar_potential
)

images = neb_path.images  # NEB 的內插結構
```

## 分子分析

### 鍵分析

```python
# 取得共價鍵
bonds = mol.get_covalent_bonds()

for bond in bonds:
    print(f"{bond.site1.species_string} - {bond.site2.species_string}: {bond.length:.2f} Å")
```

### 分子圖

```python
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

# 建構分子圖
mg = MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN())

# 取得片段
fragments = mg.get_disconnected_fragments()

# 尋找環
rings = mg.find_rings()
```

## 光譜學

### X 射線吸收光譜（XAS）

```python
from pymatgen.analysis.xas.spectrum import XAS

# 讀取 XAS 光譜
xas = XAS.from_file("xas.dat")

# 標準化和處理
xas.normalize()
```

## 其他分析工具

### 晶界

```python
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator

gb_gen = GrainBoundaryGenerator(struct)
gb_structures = gb_gen.generate_grain_boundaries(
    rotation_axis=[0, 0, 1],
    rotation_angle=36.87,  # 度
)
```

### 原型和結構匹配

```python
from pymatgen.analysis.prototypes import AflowPrototypeMatcher

matcher = AflowPrototypeMatcher()
prototype = matcher.get_prototypes(struct)
```

## 最佳實務

1. **從簡單開始**：在進階方法之前使用基本分析
2. **驗證結果**：使用多種方法交叉檢查分析
3. **考慮對稱性**：使用 `SpacegroupAnalyzer` 減少計算成本
4. **檢查收斂**：確保輸入結構已充分收斂
5. **使用適當方法**：不同分析具有不同的準確度/速度權衡
6. **視覺化結果**：使用內建繪圖器進行快速驗證
7. **儲存中間結果**：複雜分析可能耗時
