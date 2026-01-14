# RDKit API 參考

本文件提供 RDKit Python API 的完整參考，按功能組織。

## 核心模組：rdkit.Chem

處理分子的基本模組。

### 分子 I/O

**讀取分子：**

- `Chem.MolFromSmiles(smiles, sanitize=True)` - 解析 SMILES 字串
- `Chem.MolFromSmarts(smarts)` - 解析 SMARTS 模式
- `Chem.MolFromMolFile(filename, sanitize=True, removeHs=True)` - 讀取 MOL 檔案
- `Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=True)` - 解析 MOL 區塊字串
- `Chem.MolFromMol2File(filename, sanitize=True, removeHs=True)` - 讀取 MOL2 檔案
- `Chem.MolFromMol2Block(molblock, sanitize=True, removeHs=True)` - 解析 MOL2 區塊
- `Chem.MolFromPDBFile(filename, sanitize=True, removeHs=True)` - 讀取 PDB 檔案
- `Chem.MolFromPDBBlock(pdbblock, sanitize=True, removeHs=True)` - 解析 PDB 區塊
- `Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)` - 解析 InChI 字串
- `Chem.MolFromSequence(seq, sanitize=True)` - 從胜肽序列建立分子

**寫入分子：**

- `Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)` - 轉換為 SMILES
- `Chem.MolToSmarts(mol, isomericSmarts=False)` - 轉換為 SMARTS
- `Chem.MolToMolBlock(mol, includeStereo=True, confId=-1)` - 轉換為 MOL 區塊
- `Chem.MolToMolFile(mol, filename, includeStereo=True, confId=-1)` - 寫入 MOL 檔案
- `Chem.MolToPDBBlock(mol, confId=-1)` - 轉換為 PDB 區塊
- `Chem.MolToPDBFile(mol, filename, confId=-1)` - 寫入 PDB 檔案
- `Chem.MolToInchi(mol, options='')` - 轉換為 InChI
- `Chem.MolToInchiKey(mol, options='')` - 生成 InChI 鍵
- `Chem.MolToSequence(mol)` - 轉換為胜肽序列

**批次 I/O：**

- `Chem.SDMolSupplier(filename, sanitize=True, removeHs=True)` - SDF 檔案讀取器
- `Chem.ForwardSDMolSupplier(fileobj, sanitize=True, removeHs=True)` - 僅向前 SDF 讀取器
- `Chem.MultithreadedSDMolSupplier(filename, numWriterThreads=1)` - 並行 SDF 讀取器
- `Chem.SmilesMolSupplier(filename, delimiter=' ', titleLine=True)` - SMILES 檔案讀取器
- `Chem.SDWriter(filename)` - SDF 檔案寫入器
- `Chem.SmilesWriter(filename, delimiter=' ', includeHeader=True)` - SMILES 檔案寫入器

### 分子操作

**清理：**

- `Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_ALL, catchErrors=False)` - 清理分子
- `Chem.DetectChemistryProblems(mol, sanitizeOps=SANITIZE_ALL)` - 偵測清理問題
- `Chem.AssignStereochemistry(mol, cleanIt=True, force=False)` - 分配立體化學
- `Chem.FindPotentialStereo(mol)` - 尋找潛在立體中心
- `Chem.AssignStereochemistryFrom3D(mol, confId=-1)` - 從 3D 座標分配立體

**氫管理：**

- `Chem.AddHs(mol, explicitOnly=False, addCoords=False)` - 新增顯式氫
- `Chem.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False)` - 移除氫
- `Chem.RemoveAllHs(mol)` - 移除所有氫

**芳香性：**

- `Chem.SetAromaticity(mol, model=AROMATICITY_RDKIT)` - 設定芳香性模型
- `Chem.Kekulize(mol, clearAromaticFlags=False)` - Kekulize 芳香鍵
- `Chem.SetConjugation(mol)` - 設定共軛標記

**片段：**

- `Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=True)` - 取得斷開的片段
- `Chem.FragmentOnBonds(mol, bondIndices, addDummies=True)` - 在特定鍵上片段化
- `Chem.ReplaceSubstructs(mol, query, replacement, replaceAll=False)` - 取代子結構
- `Chem.DeleteSubstructs(mol, query, onlyFrags=False)` - 刪除子結構

**立體化學：**

- `Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)` - 尋找手性中心
- `Chem.FindPotentialStereo(mol, cleanIt=True)` - 尋找潛在立體中心

### 子結構搜尋

**基本匹配：**

- `mol.HasSubstructMatch(query, useChirality=False)` - 檢查子結構匹配
- `mol.GetSubstructMatch(query, useChirality=False)` - 取得第一個匹配
- `mol.GetSubstructMatches(query, uniquify=True, useChirality=False)` - 取得所有匹配
- `mol.GetSubstructMatches(query, maxMatches=1000)` - 限制匹配數量

### 分子性質

**原子方法：**

- `atom.GetSymbol()` - 原子符號
- `atom.GetAtomicNum()` - 原子序數
- `atom.GetDegree()` - 鍵數量
- `atom.GetTotalDegree()` - 包含氫
- `atom.GetFormalCharge()` - 形式電荷
- `atom.GetNumRadicalElectrons()` - 自由基電子
- `atom.GetIsAromatic()` - 芳香性標記
- `atom.GetHybridization()` - 雜化（SP、SP2、SP3 等）
- `atom.GetIdx()` - 原子索引
- `atom.IsInRing()` - 在任意環中
- `atom.IsInRingSize(size)` - 在特定大小的環中
- `atom.GetChiralTag()` - 手性標籤

**鍵方法：**

- `bond.GetBondType()` - 鍵類型（SINGLE、DOUBLE、TRIPLE、AROMATIC）
- `bond.GetBeginAtomIdx()` - 起始原子索引
- `bond.GetEndAtomIdx()` - 結束原子索引
- `bond.GetIsConjugated()` - 共軛標記
- `bond.GetIsAromatic()` - 芳香性標記
- `bond.IsInRing()` - 在任意環中
- `bond.GetStereo()` - 立體化學（STEREONONE、STEREOZ、STEREOE 等）

**分子方法：**

- `mol.GetNumAtoms(onlyExplicit=True)` - 原子數量
- `mol.GetNumHeavyAtoms()` - 重原子數量
- `mol.GetNumBonds()` - 鍵數量
- `mol.GetAtoms()` - 原子迭代器
- `mol.GetBonds()` - 鍵迭代器
- `mol.GetAtomWithIdx(idx)` - 取得特定原子
- `mol.GetBondWithIdx(idx)` - 取得特定鍵
- `mol.GetRingInfo()` - 環資訊物件

**環資訊：**

- `Chem.GetSymmSSSR(mol)` - 取得最小最小環集合
- `Chem.GetSSSR(mol)` - GetSymmSSSR 的別名
- `ring_info.NumRings()` - 環數量
- `ring_info.AtomRings()` - 環中原子索引的元組
- `ring_info.BondRings()` - 環中鍵索引的元組

## rdkit.Chem.AllChem

擴展化學功能。

### 2D/3D 座標生成

- `AllChem.Compute2DCoords(mol, canonOrient=True, clearConfs=True)` - 生成 2D 座標
- `AllChem.EmbedMolecule(mol, maxAttempts=0, randomSeed=-1, useRandomCoords=False)` - 生成 3D 構象異構體
- `AllChem.EmbedMultipleConfs(mol, numConfs=10, maxAttempts=0, randomSeed=-1)` - 生成多個構象異構體
- `AllChem.ConstrainedEmbed(mol, core, useTethers=True)` - 約束嵌入
- `AllChem.GenerateDepictionMatching2DStructure(mol, reference, refPattern=None)` - 對齊到模板

### 力場優化

- `AllChem.UFFOptimizeMolecule(mol, maxIters=200, confId=-1)` - UFF 優化
- `AllChem.MMFFOptimizeMolecule(mol, maxIters=200, confId=-1, mmffVariant='MMFF94')` - MMFF 優化
- `AllChem.UFFGetMoleculeForceField(mol, confId=-1)` - 取得 UFF 力場物件
- `AllChem.MMFFGetMoleculeForceField(mol, pyMMFFMolProperties, confId=-1)` - 取得 MMFF 力場

### 構象異構體分析

- `AllChem.GetConformerRMS(mol, confId1, confId2, prealigned=False)` - 計算 RMSD
- `AllChem.GetConformerRMSMatrix(mol, prealigned=False)` - RMSD 矩陣
- `AllChem.AlignMol(prbMol, refMol, prbCid=-1, refCid=-1)` - 對齊分子
- `AllChem.AlignMolConformers(mol)` - 對齊所有構象異構體

### 反應

- `AllChem.ReactionFromSmarts(smarts, useSmiles=False)` - 從 SMARTS 建立反應
- `reaction.RunReactants(reactants)` - 應用反應
- `reaction.RunReactant(reactant, reactionIdx)` - 應用到特定反應物
- `AllChem.CreateDifferenceFingerprintForReaction(reaction)` - 反應指紋

### 指紋

- `AllChem.GetMorganFingerprint(mol, radius, useFeatures=False)` - Morgan 指紋
- `AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)` - Morgan 位元向量
- `AllChem.GetHashedMorganFingerprint(mol, radius, nBits=2048)` - 雜湊 Morgan
- `AllChem.GetErGFingerprint(mol)` - ErG 指紋

## rdkit.Chem.Descriptors

分子描述子計算。

### 常見描述子

- `Descriptors.MolWt(mol)` - 分子量
- `Descriptors.ExactMolWt(mol)` - 精確分子量
- `Descriptors.HeavyAtomMolWt(mol)` - 重原子分子量
- `Descriptors.MolLogP(mol)` - LogP（親脂性）
- `Descriptors.MolMR(mol)` - 莫耳折射率
- `Descriptors.TPSA(mol)` - 拓撲極性表面積
- `Descriptors.NumHDonors(mol)` - 氫鍵供體
- `Descriptors.NumHAcceptors(mol)` - 氫鍵受體
- `Descriptors.NumRotatableBonds(mol)` - 可旋轉鍵
- `Descriptors.NumAromaticRings(mol)` - 芳香環
- `Descriptors.NumSaturatedRings(mol)` - 飽和環
- `Descriptors.NumAliphaticRings(mol)` - 脂肪族環
- `Descriptors.NumAromaticHeterocycles(mol)` - 芳香雜環
- `Descriptors.NumRadicalElectrons(mol)` - 自由基電子
- `Descriptors.NumValenceElectrons(mol)` - 價電子

### 批次計算

- `Descriptors.CalcMolDescriptors(mol)` - 計算所有描述子為字典

### 描述子列表

- `Descriptors._descList` - 所有描述子的 (名稱, 函數) 元組列表

## rdkit.Chem.Draw

分子視覺化。

### 圖像生成

- `Draw.MolToImage(mol, size=(300,300), kekulize=True, wedgeBonds=True, highlightAtoms=None)` - 生成 PIL 圖像
- `Draw.MolToFile(mol, filename, size=(300,300), kekulize=True, wedgeBonds=True)` - 儲存到檔案
- `Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200,200), legends=None)` - 分子網格
- `Draw.MolsMatrixToGridImage(mols, molsPerRow=3, subImgSize=(200,200), legends=None)` - 巢狀網格
- `Draw.ReactionToImage(rxn, subImgSize=(200,200))` - 反應圖像

### 指紋視覺化

- `Draw.DrawMorganBit(mol, bitId, bitInfo, whichExample=0)` - 視覺化 Morgan 位元
- `Draw.DrawMorganBits(bits, mol, bitInfo, molsPerRow=3)` - 多個 Morgan 位元
- `Draw.DrawRDKitBit(mol, bitId, bitInfo, whichExample=0)` - 視覺化 RDKit 位元

### IPython 整合

- `Draw.IPythonConsole` - Jupyter 整合模組
- `Draw.IPythonConsole.ipython_useSVG` - 使用 SVG (True) 或 PNG (False)
- `Draw.IPythonConsole.molSize` - 預設分子圖像大小

### 繪圖選項

- `rdMolDraw2D.MolDrawOptions()` - 取得繪圖選項物件
  - `.addAtomIndices` - 顯示原子索引
  - `.addBondIndices` - 顯示鍵索引
  - `.addStereoAnnotation` - 顯示立體化學
  - `.bondLineWidth` - 線寬
  - `.highlightBondWidthMultiplier` - 標記寬度
  - `.minFontSize` - 最小字體大小
  - `.maxFontSize` - 最大字體大小

## rdkit.Chem.rdMolDescriptors

額外描述子計算。

- `rdMolDescriptors.CalcNumRings(mol)` - 環數量
- `rdMolDescriptors.CalcNumAromaticRings(mol)` - 芳香環
- `rdMolDescriptors.CalcNumAliphaticRings(mol)` - 脂肪族環
- `rdMolDescriptors.CalcNumSaturatedRings(mol)` - 飽和環
- `rdMolDescriptors.CalcNumHeterocycles(mol)` - 雜環
- `rdMolDescriptors.CalcNumAromaticHeterocycles(mol)` - 芳香雜環
- `rdMolDescriptors.CalcNumSpiroAtoms(mol)` - 螺原子
- `rdMolDescriptors.CalcNumBridgeheadAtoms(mol)` - 橋頭原子
- `rdMolDescriptors.CalcFractionCsp3(mol)` - sp3 碳比例
- `rdMolDescriptors.CalcLabuteASA(mol)` - Labute 可及表面積
- `rdMolDescriptors.CalcTPSA(mol)` - TPSA
- `rdMolDescriptors.CalcMolFormula(mol)` - 分子式

## rdkit.Chem.Scaffolds

骨架分析。

### Murcko 骨架

- `MurckoScaffold.GetScaffoldForMol(mol)` - 取得 Murcko 骨架
- `MurckoScaffold.MakeScaffoldGeneric(mol)` - 通用骨架
- `MurckoScaffold.MurckoDecompose(mol)` - 分解為骨架和側鏈

## rdkit.Chem.rdMolHash

分子雜湊和標準化。

- `rdMolHash.MolHash(mol, hashFunction)` - 生成雜湊
  - `rdMolHash.HashFunction.AnonymousGraph` - 匿名化結構
  - `rdMolHash.HashFunction.CanonicalSmiles` - 規範 SMILES
  - `rdMolHash.HashFunction.ElementGraph` - 元素圖
  - `rdMolHash.HashFunction.MurckoScaffold` - Murcko 骨架
  - `rdMolHash.HashFunction.Regioisomer` - 區域異構體（無立體）
  - `rdMolHash.HashFunction.NetCharge` - 淨電荷
  - `rdMolHash.HashFunction.HetAtomProtomer` - 雜原子質子體
  - `rdMolHash.HashFunction.HetAtomTautomer` - 雜原子互變異構體

## rdkit.Chem.MolStandardize

分子標準化。

- `rdMolStandardize.Normalize(mol)` - 標準化官能基
- `rdMolStandardize.Reionize(mol)` - 修正離子化狀態
- `rdMolStandardize.RemoveFragments(mol)` - 移除小片段
- `rdMolStandardize.Cleanup(mol)` - 完整清理（標準化 + 重新離子化 + 移除）
- `rdMolStandardize.Uncharger()` - 建立去電荷器物件
  - `.uncharge(mol)` - 移除電荷
- `rdMolStandardize.TautomerEnumerator()` - 列舉互變異構體
  - `.Enumerate(mol)` - 生成互變異構體
  - `.Canonicalize(mol)` - 取得規範互變異構體

## rdkit.DataStructs

指紋相似性和操作。

### 相似性度量

- `DataStructs.TanimotoSimilarity(fp1, fp2)` - Tanimoto 係數
- `DataStructs.DiceSimilarity(fp1, fp2)` - Dice 係數
- `DataStructs.CosineSimilarity(fp1, fp2)` - 餘弦相似性
- `DataStructs.SokalSimilarity(fp1, fp2)` - Sokal 相似性
- `DataStructs.KulczynskiSimilarity(fp1, fp2)` - Kulczynski 相似性
- `DataStructs.McConnaugheySimilarity(fp1, fp2)` - McConnaughey 相似性

### 批次操作

- `DataStructs.BulkTanimotoSimilarity(fp, fps)` - 指紋列表的 Tanimoto
- `DataStructs.BulkDiceSimilarity(fp, fps)` - 列表的 Dice
- `DataStructs.BulkCosineSimilarity(fp, fps)` - 列表的餘弦

### 距離度量

- `DataStructs.TanimotoDistance(fp1, fp2)` - 1 - Tanimoto
- `DataStructs.DiceDistance(fp1, fp2)` - 1 - Dice

## rdkit.Chem.AtomPairs

原子對指紋。

- `Pairs.GetAtomPairFingerprint(mol, minLength=1, maxLength=30)` - 原子對指紋
- `Pairs.GetAtomPairFingerprintAsBitVect(mol, minLength=1, maxLength=30, nBits=2048)` - 作為位元向量
- `Pairs.GetHashedAtomPairFingerprint(mol, nBits=2048, minLength=1, maxLength=30)` - 雜湊版本

## rdkit.Chem.Torsions

拓撲扭轉角指紋。

- `Torsions.GetTopologicalTorsionFingerprint(mol, targetSize=4)` - 扭轉角指紋
- `Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol, targetSize=4)` - 作為整數向量
- `Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits=2048, targetSize=4)` - 雜湊版本

## rdkit.Chem.MACCSkeys

MACCS 結構鍵。

- `MACCSkeys.GenMACCSKeys(mol)` - 生成 166 位元 MACCS 鍵

## rdkit.Chem.ChemicalFeatures

藥效團特徵。

- `ChemicalFeatures.BuildFeatureFactory(featureFile)` - 建立特徵工廠
- `factory.GetFeaturesForMol(mol)` - 取得藥效團特徵
- `feature.GetFamily()` - 特徵家族（Donor、Acceptor 等）
- `feature.GetType()` - 特徵類型
- `feature.GetAtomIds()` - 涉及特徵的原子

## rdkit.ML.Cluster.Butina

群集演算法。

- `Butina.ClusterData(distances, nPts, distThresh, isDistData=True)` - Butina 群集
  - 返回包含群集成員的元組的元組

## rdkit.Chem.rdFingerprintGenerator

現代指紋生成 API（RDKit 2020.09+）。

- `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` - Morgan 生成器
- `rdFingerprintGenerator.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=2048)` - RDKit FP 生成器
- `rdFingerprintGenerator.GetAtomPairGenerator(minDistance=1, maxDistance=30)` - 原子對生成器
- `generator.GetFingerprint(mol)` - 生成指紋
- `generator.GetCountFingerprint(mol)` - 計數型指紋

## 常見參數

### 清理操作

- `SANITIZE_NONE` - 不清理
- `SANITIZE_ALL` - 所有操作（預設）
- `SANITIZE_CLEANUP` - 基本清理
- `SANITIZE_PROPERTIES` - 計算性質
- `SANITIZE_SYMMRINGS` - 對稱化環
- `SANITIZE_KEKULIZE` - Kekulize 芳香環
- `SANITIZE_FINDRADICALS` - 尋找自由基電子
- `SANITIZE_SETAROMATICITY` - 設定芳香性
- `SANITIZE_SETCONJUGATION` - 設定共軛
- `SANITIZE_SETHYBRIDIZATION` - 設定雜化
- `SANITIZE_CLEANUPCHIRALITY` - 清理手性

### 鍵類型

- `BondType.SINGLE` - 單鍵
- `BondType.DOUBLE` - 雙鍵
- `BondType.TRIPLE` - 參鍵
- `BondType.AROMATIC` - 芳香鍵
- `BondType.DATIVE` - 配位鍵
- `BondType.UNSPECIFIED` - 未指定

### 雜化

- `HybridizationType.S` - S
- `HybridizationType.SP` - SP
- `HybridizationType.SP2` - SP2
- `HybridizationType.SP3` - SP3
- `HybridizationType.SP3D` - SP3D
- `HybridizationType.SP3D2` - SP3D2

### 手性

- `ChiralType.CHI_UNSPECIFIED` - 未指定
- `ChiralType.CHI_TETRAHEDRAL_CW` - 順時針
- `ChiralType.CHI_TETRAHEDRAL_CCW` - 逆時針

## 安裝

```bash
# 使用 conda（推薦）
conda install -c conda-forge rdkit

# 使用 pip
pip install rdkit-pypi
```

## 匯入

```python
# 核心功能
from rdkit import Chem
from rdkit.Chem import AllChem

# 描述子
from rdkit.Chem import Descriptors

# 繪圖
from rdkit.Chem import Draw

# 相似性
from rdkit import DataStructs
```
