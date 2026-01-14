# Pymatgen I/O 和檔案格式參考

本參考記錄 pymatgen 廣泛的輸入/輸出功能，用於讀寫跨 100+ 種檔案格式的結構和計算資料。

## 一般 I/O 理念

Pymatgen 透過 `from_file()` 和 `to()` 方法提供統一的檔案操作介面，具有基於副檔名的自動格式偵測。

### 讀取檔案

```python
from pymatgen.core import Structure, Molecule

# 自動格式偵測
struct = Structure.from_file("POSCAR")
struct = Structure.from_file("structure.cif")
mol = Molecule.from_file("molecule.xyz")

# 明確格式指定
struct = Structure.from_file("file.txt", fmt="cif")
```

### 寫入檔案

```python
# 寫入檔案（從副檔名推斷格式）
struct.to(filename="output.cif")
struct.to(filename="POSCAR")
struct.to(filename="structure.xyz")

# 取得字串表示而不寫入
cif_string = struct.to(fmt="cif")
poscar_string = struct.to(fmt="poscar")
```

## 結構檔案格式

### CIF（晶體學資訊檔案）
晶體學資料的標準格式。

```python
from pymatgen.io.cif import CifParser, CifWriter

# 讀取
parser = CifParser("structure.cif")
structure = parser.get_structures()[0]  # 回傳結構列表

# 寫入
writer = CifWriter(struct)
writer.write_file("output.cif")

# 或使用便捷方法
struct = Structure.from_file("structure.cif")
struct.to(filename="output.cif")
```

**主要功能：**
- 支援對稱性資訊
- 可包含多個結構
- 保留空間群和對稱操作
- 處理部分佔據

### POSCAR/CONTCAR（VASP）
VASP 的結構格式。

```python
from pymatgen.io.vasp import Poscar

# 讀取
poscar = Poscar.from_file("POSCAR")
structure = poscar.structure

# 寫入
poscar = Poscar(struct)
poscar.write_file("POSCAR")

# 或使用便捷方法
struct = Structure.from_file("POSCAR")
struct.to(filename="POSCAR")
```

**主要功能：**
- 支援選擇性動力學
- 可包含速度（XDATCAR 格式）
- 保留晶格和座標精度

### XYZ
簡單的分子座標格式。

```python
# 用於分子
mol = Molecule.from_file("molecule.xyz")
mol.to(filename="output.xyz")

# 用於結構（笛卡爾座標）
struct.to(filename="structure.xyz")
```

### PDB（蛋白質資料庫）
生物分子的常見格式。

```python
mol = Molecule.from_file("protein.pdb")
mol.to(filename="output.pdb")
```

### JSON/YAML
透過字典序列化。

```python
import json
import yaml

# JSON
with open("structure.json", "w") as f:
    json.dump(struct.as_dict(), f)

with open("structure.json", "r") as f:
    struct = Structure.from_dict(json.load(f))

# YAML
with open("structure.yaml", "w") as f:
    yaml.dump(struct.as_dict(), f)

with open("structure.yaml", "r") as f:
    struct = Structure.from_dict(yaml.safe_load(f))
```

## 電子結構程式碼 I/O

### VASP

pymatgen 中最全面的整合。

#### 輸入檔案

```python
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints, VaspInput

# INCAR（計算參數）
incar = Incar.from_file("INCAR")
incar = Incar({"ENCUT": 520, "ISMEAR": 0, "SIGMA": 0.05})
incar.write_file("INCAR")

# KPOINTS（k 點網格）
from pymatgen.io.vasp.inputs import Kpoints
kpoints = Kpoints.automatic(20)  # 20x20x20 Gamma 中心網格
kpoints = Kpoints.automatic_density(struct, 1000)  # 按密度
kpoints.write_file("KPOINTS")

# POTCAR（贗勢）
potcar = Potcar(["Fe_pv", "O"])  # 指定泛函變體

# 完整輸入集
vasp_input = VaspInput(incar, kpoints, poscar, potcar)
vasp_input.write_input("./vasp_calc")
```

#### 輸出檔案

```python
from pymatgen.io.vasp.outputs import Vasprun, Outcar, Oszicar, Eigenval

# vasprun.xml（全面輸出）
vasprun = Vasprun("vasprun.xml")
final_structure = vasprun.final_structure
energy = vasprun.final_energy
band_structure = vasprun.get_band_structure()
dos = vasprun.complete_dos

# OUTCAR
outcar = Outcar("OUTCAR")
magnetization = outcar.total_mag
elastic_tensor = outcar.elastic_tensor

# OSZICAR（收斂資訊）
oszicar = Oszicar("OSZICAR")
```

#### 輸入集

Pymatgen 為常見計算提供預配置的輸入集：

```python
from pymatgen.io.vasp.sets import (
    MPRelaxSet,      # Materials Project 弛豫
    MPStaticSet,     # 靜態計算
    MPNonSCFSet,     # 非自洽（能帶結構）
    MPSOCSet,        # 自旋軌道耦合
    MPHSERelaxSet,   # HSE06 混合泛函
)

# 建立輸入集
relax = MPRelaxSet(struct)
relax.write_input("./relax_calc")

# 自訂參數
static = MPStaticSet(struct, user_incar_settings={"ENCUT": 600})
static.write_input("./static_calc")
```

### Gaussian

量子化學套件整合。

```python
from pymatgen.io.gaussian import GaussianInput, GaussianOutput

# 輸入
gin = GaussianInput(
    mol,
    charge=0,
    spin_multiplicity=1,
    functional="B3LYP",
    basis_set="6-31G(d)",
    route_parameters={"Opt": None, "Freq": None}
)
gin.write_file("input.gjf")

# 輸出
gout = GaussianOutput("output.log")
final_mol = gout.final_structure
energy = gout.final_energy
frequencies = gout.frequencies
```

### LAMMPS

經典分子動力學。

```python
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsInputFile

# 結構轉 LAMMPS 資料檔案
lammps_data = LammpsData.from_structure(struct)
lammps_data.write_file("data.lammps")

# LAMMPS 輸入腳本
lammps_input = LammpsInputFile.from_file("in.lammps")
```

### Quantum ESPRESSO

```python
from pymatgen.io.pwscf import PWInput, PWOutput

# 輸入
pwin = PWInput(
    struct,
    control={"calculation": "scf"},
    system={"ecutwfc": 50, "ecutrho": 400},
    electrons={"conv_thr": 1e-8}
)
pwin.write_file("pw.in")

# 輸出
pwout = PWOutput("pw.out")
final_structure = pwout.final_structure
energy = pwout.final_energy
```

### ABINIT

```python
from pymatgen.io.abinit import AbinitInput

abin = AbinitInput(struct, pseudos)
abin.set_vars(ecut=10, nband=10)
abin.write("abinit.in")
```

### CP2K

```python
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.outputs import Cp2kOutput

# 輸入
cp2k_input = Cp2kInput.from_file("cp2k.inp")

# 輸出
cp2k_output = Cp2kOutput("cp2k.out")
```

### FEFF（XAS/XANES）

```python
from pymatgen.io.feff import FeffInput

feff_input = FeffInput(struct, absorbing_atom="Fe")
feff_input.write_file("feff.inp")
```

### LMTO（Stuttgart TB-LMTO-ASA）

```python
from pymatgen.io.lmto import LMTOCtrl

ctrl = LMTOCtrl.from_file("CTRL")
ctrl.structure
```

### Q-Chem

```python
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.outputs import QCOutput

# 輸入
qc_input = QCInput(
    mol,
    rem={"method": "B3LYP", "basis": "6-31G*", "job_type": "opt"}
)
qc_input.write_file("mol.qin")

# 輸出
qc_output = QCOutput("mol.qout")
```

### Exciting

```python
from pymatgen.io.exciting import ExcitingInput

exc_input = ExcitingInput(struct)
exc_input.write_file("input.xml")
```

### ATAT（合金理論自動化工具包）

```python
from pymatgen.io.atat import Mcsqs

mcsqs = Mcsqs(struct)
mcsqs.write_input(".")
```

## 特殊用途格式

### Phonopy

```python
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

# 轉換為 phonopy 結構
phonopy_struct = get_phonopy_structure(struct)

# 從 phonopy 轉換
struct = get_pmg_structure(phonopy_struct)
```

### ASE（原子模擬環境）

```python
from pymatgen.io.ase import AseAtomsAdaptor

adaptor = AseAtomsAdaptor()

# Pymatgen 轉 ASE
atoms = adaptor.get_atoms(struct)

# ASE 轉 Pymatgen
struct = adaptor.get_structure(atoms)
```

### Zeo++（多孔材料）

```python
from pymatgen.io.zeopp import get_voronoi_nodes, get_high_accuracy_voronoi_nodes

# 分析孔洞結構
vor_nodes = get_voronoi_nodes(struct)
```

### BabelMolAdaptor（OpenBabel）

```python
from pymatgen.io.babel import BabelMolAdaptor

adaptor = BabelMolAdaptor(mol)

# 轉換為不同格式
pdb_str = adaptor.pdbstring
sdf_str = adaptor.write_file("mol.sdf", file_format="sdf")

# 生成 3D 座標
adaptor.add_hydrogen()
adaptor.make3d()
```

## 煉金術和轉換 I/O

### TransformedStructure

追蹤轉換歷史的結構。

```python
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import (
    SupercellTransformation,
    SubstitutionTransformation
)

# 建立轉換結構
ts = TransformedStructure(struct, [])
ts.append_transformation(SupercellTransformation([[2,0,0],[0,2,0],[0,0,2]]))
ts.append_transformation(SubstitutionTransformation({"Fe": "Mn"}))

# 寫入包含歷史
ts.write_vasp_input("./calc_dir")

# 從 SNL（結構筆記本語言）讀取
ts = TransformedStructure.from_snl(snl)
```

## 批次操作

### CifTransmuter

處理多個 CIF 檔案。

```python
from pymatgen.alchemy.transmuters import CifTransmuter

transmuter = CifTransmuter.from_filenames(
    ["structure1.cif", "structure2.cif"],
    [SupercellTransformation([[2,0,0],[0,2,0],[0,0,2]])]
)

# 寫入所有結構
transmuter.write_vasp_input("./batch_calc")
```

### PoscarTransmuter

類似用於 POSCAR 檔案。

```python
from pymatgen.alchemy.transmuters import PoscarTransmuter

transmuter = PoscarTransmuter.from_filenames(
    ["POSCAR1", "POSCAR2"],
    [transformation1, transformation2]
)
```

## 最佳實務

1. **自動格式偵測**：盡可能使用 `from_file()` 和 `to()` 方法
2. **錯誤處理**：始終將檔案 I/O 包裝在 try-except 區塊中
3. **格式專用解析器**：使用專門解析器（例如 `Vasprun`）進行詳細輸出分析
4. **輸入集**：偏好預配置輸入集而非手動參數指定
5. **序列化**：使用 JSON/YAML 進行長期儲存和版本控制
6. **批次處理**：使用 transmuter 將轉換應用於多個結構

## 支援格式摘要

### 結構格式：
CIF、POSCAR/CONTCAR、XYZ、PDB、XSF、PWMAT、Res、CSSR、JSON、YAML

### 電子結構程式碼：
VASP、Gaussian、LAMMPS、Quantum ESPRESSO、ABINIT、CP2K、FEFF、Q-Chem、LMTO、Exciting、NWChem、AIMS、晶體學資料格式

### 分子格式：
XYZ、PDB、MOL、SDF、PQR，透過 OpenBabel（許多額外格式）

### 特殊用途：
Phonopy、ASE、Zeo++、Lobster、BoltzTraP
