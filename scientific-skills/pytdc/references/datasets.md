# TDC 資料集完整目錄

本文件提供 Therapeutics Data Commons 中所有可用資料集的完整目錄，按任務類別組織。

## 單一實例預測資料集

### ADME（吸收、分布、代謝、排泄）

**吸收：**
- `Caco2_Wang` - Caco-2 細胞通透性（906 個化合物）
- `Caco2_AstraZeneca` - 來自 AstraZeneca 的 Caco-2 通透性（700 個化合物）
- `HIA_Hou` - 人類腸道吸收（578 個化合物）
- `Pgp_Broccatelli` - P-醣蛋白抑制（1,212 個化合物）
- `Bioavailability_Ma` - 口服生物利用度（640 個化合物）
- `F20_edrug3d` - 口服生物利用度 F>=20%（1,017 個化合物）
- `F30_edrug3d` - 口服生物利用度 F>=30%（1,017 個化合物）

**分布：**
- `BBB_Martins` - 血腦屏障穿透（1,975 個化合物）
- `PPBR_AZ` - 血漿蛋白結合率（1,797 個化合物）
- `VDss_Lombardo` - 穩態分布體積（1,130 個化合物）

**代謝：**
- `CYP2C19_Veith` - CYP2C19 抑制（12,665 個化合物）
- `CYP2D6_Veith` - CYP2D6 抑制（13,130 個化合物）
- `CYP3A4_Veith` - CYP3A4 抑制（12,328 個化合物）
- `CYP1A2_Veith` - CYP1A2 抑制（12,579 個化合物）
- `CYP2C9_Veith` - CYP2C9 抑制（12,092 個化合物）
- `CYP2C9_Substrate_CarbonMangels` - CYP2C9 受質（666 個化合物）
- `CYP2D6_Substrate_CarbonMangels` - CYP2D6 受質（664 個化合物）
- `CYP3A4_Substrate_CarbonMangels` - CYP3A4 受質（667 個化合物）

**排泄：**
- `Half_Life_Obach` - 半衰期（667 個化合物）
- `Clearance_Hepatocyte_AZ` - 肝細胞清除率（1,020 個化合物）
- `Clearance_Microsome_AZ` - 微粒體清除率（1,102 個化合物）

**溶解度與親脂性：**
- `Solubility_AqSolDB` - 水溶性（9,982 個化合物）
- `Lipophilicity_AstraZeneca` - 親脂性（logD）（4,200 個化合物）
- `HydrationFreeEnergy_FreeSolv` - 水合自由能（642 個化合物）

### 毒性

**器官毒性：**
- `hERG` - hERG 通道抑制/心臟毒性（648 個化合物）
- `hERG_Karim` - hERG 阻斷劑擴展資料集（13,445 個化合物）
- `DILI` - 藥物誘導性肝損傷（475 個化合物）
- `Skin_Reaction` - 皮膚反應（404 個化合物）
- `Carcinogens_Lagunin` - 致癌性（278 個化合物）
- `Respiratory_Toxicity` - 呼吸道毒性（278 個化合物）

**一般毒性：**
- `AMES` - Ames 致突變性（7,255 個化合物）
- `LD50_Zhu` - 急性毒性 LD50（7,385 個化合物）
- `ClinTox` - 臨床試驗毒性（1,478 個化合物）
- `SkinSensitization` - 皮膚致敏性（278 個化合物）
- `EyeCorrosion` - 眼部腐蝕性（278 個化合物）
- `EyeIrritation` - 眼部刺激性（278 個化合物）

**環境毒性：**
- `Tox21-AhR` - 核受體訊號傳導（8,169 個化合物）
- `Tox21-AR` - 雄激素受體（9,362 個化合物）
- `Tox21-AR-LBD` - 雄激素受體配體結合（8,343 個化合物）
- `Tox21-ARE` - 抗氧化反應元件（6,475 個化合物）
- `Tox21-aromatase` - 芳香酶抑制（6,733 個化合物）
- `Tox21-ATAD5` - DNA 損傷（8,163 個化合物）
- `Tox21-ER` - 雌激素受體（7,257 個化合物）
- `Tox21-ER-LBD` - 雌激素受體配體結合（8,163 個化合物）
- `Tox21-HSE` - 熱休克反應（8,162 個化合物）
- `Tox21-MMP` - 粒線體膜電位（7,394 個化合物）
- `Tox21-p53` - p53 路徑（8,163 個化合物）
- `Tox21-PPAR-gamma` - PPAR gamma 活化（7,396 個化合物）

### HTS（高通量篩選）

**SARS-CoV-2：**
- `SARSCoV2_Vitro_Touret` - 體外抗病毒活性（1,484 個化合物）
- `SARSCoV2_3CLPro_Diamond` - 3CL 蛋白酶抑制（879 個化合物）
- `SARSCoV2_Vitro_AlabdulKareem` - 體外篩選（5,953 個化合物）

**其他標靶：**
- `Orexin1_Receptor_Butkiewicz` - 食慾素受體篩選（4,675 個化合物）
- `M1_Receptor_Agonist_Butkiewicz` - M1 受體激動劑（1,700 個化合物）
- `M1_Receptor_Antagonist_Butkiewicz` - M1 受體拮抗劑（1,700 個化合物）
- `HIV_Butkiewicz` - HIV 抑制（40,000+ 個化合物）
- `ToxCast` - 環境化學品篩選（8,597 個化合物）

### QM（量子力學）

- `QM7` - 量子力學屬性（7,160 個分子）
- `QM8` - 電子光譜和激發態（21,786 個分子）
- `QM9` - 幾何、能量、電子、熱力學屬性（133,885 個分子）

### Yields（產率）

- `Buchwald-Hartwig` - 反應產率預測（3,955 個反應）
- `USPTO_Yields` - 來自 USPTO 的產率預測（853,879 個反應）

### Epitope（抗原表位）

- `IEDBpep-DiseaseBinder` - 疾病相關抗原表位結合（6,080 個肽）
- `IEDBpep-NonBinder` - 非結合肽（24,320 個肽）

### Develop（開發）

- `Manufacturing` - 製造成功預測
- `Formulation` - 製劑穩定性

### CRISPROutcome

- `CRISPROutcome_Doench` - 基因編輯效率預測（5,310 個引導 RNA）

## 多實例預測資料集

### DTI（藥物-標靶互動）

**結合親和力：**
- `BindingDB_Kd` - 解離常數（52,284 對，10,665 種藥物，1,413 個蛋白質）
- `BindingDB_IC50` - 半最大抑制濃度（991,486 對，549,205 種藥物，5,078 個蛋白質）
- `BindingDB_Ki` - 抑制常數（375,032 對，174,662 種藥物，3,070 個蛋白質）

**激酶結合：**
- `DAVIS` - Davis 激酶結合資料集（30,056 對，68 種藥物，442 個蛋白質）
- `KIBA` - KIBA 激酶結合資料集（118,254 對，2,111 種藥物，229 個蛋白質）

**二元互動：**
- `BindingDB_Patent` - 專利衍生 DTI（8,503 對）
- `BindingDB_Approval` - FDA 批准藥物 DTI（1,649 對）

### DDI（藥物-藥物互動）

- `DrugBank` - 藥物-藥物互動（191,808 對，1,706 種藥物）
- `TWOSIDES` - 基於副作用的 DDI（4,649,441 對，645 種藥物）

### PPI（蛋白質-蛋白質互動）

- `HuRI` - 人類參考蛋白質互動組（52,569 個互動）
- `STRING` - 蛋白質功能關聯（19,247 個互動）

### GDA（基因-疾病關聯）

- `DisGeNET` - 基因-疾病關聯（81,746 對）
- `PrimeKG_GDA` - 來自 PrimeKG 知識圖譜的基因-疾病

### DrugRes（藥物反應/抗性）

- `GDSC1` - 癌症藥物敏感性基因組學 v1（178,000 對）
- `GDSC2` - 癌症藥物敏感性基因組學 v2（125,000 對）

### DrugSyn（藥物協同作用）

- `DrugComb` - 藥物組合協同作用（345,502 個組合）
- `DrugCombDB` - 藥物組合資料庫（448,555 個組合）
- `OncoPolyPharmacology` - 腫瘤學藥物組合（22,737 個組合）

### PeptideMHC

- `MHC1_NetMHCpan` - MHC I 類結合（184,983 對）
- `MHC2_NetMHCIIpan` - MHC II 類結合（134,281 對）

### AntibodyAff（抗體親和力）

- `Protein_SAbDab` - 抗體-抗原親和力（1,500+ 對）

### MTI（miRNA-標靶互動）

- `miRTarBase` - 實驗驗證的 miRNA-標靶互動（380,639 對）

### Catalyst（催化劑）

- `USPTO_Catalyst` - 反應催化劑預測（11,000+ 個反應）

### TrialOutcome（試驗結果）

- `TrialOutcome_WuXi` - 臨床試驗結果預測（3,769 個試驗）

## 生成資料集

### MolGen（分子生成）

- `ChEMBL_V29` - 來自 ChEMBL 的類藥分子（1,941,410 個分子）
- `ZINC` - ZINC 資料庫子集（100,000+ 個分子）
- `GuacaMol` - 目標導向基準分子
- `Moses` - 分子集基準（1,936,962 個分子）

### RetroSyn（逆合成）

- `USPTO` - 來自 USPTO 專利的逆合成（1,939,253 個反應）
- `USPTO-50K` - 策劃的 USPTO 子集（50,000 個反應）

### PairMolGen（配對分子生成）

- `Prodrug` - 前藥到藥物轉換（1,000+ 對）
- `Metabolite` - 藥物到代謝物轉換

## 使用 retrieve_dataset_names

以程式方式存取特定任務的所有可用資料集：

```python
from tdc.utils import retrieve_dataset_names

# 取得特定任務的所有資料集
adme_datasets = retrieve_dataset_names('ADME')
tox_datasets = retrieve_dataset_names('Tox')
dti_datasets = retrieve_dataset_names('DTI')
hts_datasets = retrieve_dataset_names('HTS')
```

## 資料集統計

直接存取資料集統計：

```python
from tdc.single_pred import ADME
data = ADME(name='Caco2_Wang')

# 列印基本統計
data.print_stats()

# 取得標籤分布
data.label_distribution()
```

## 載入資料集

所有資料集遵循相同的載入模式：

```python
from tdc.<problem_type> import <TaskType>
data = <TaskType>(name='<DatasetName>')

# 取得完整資料集
df = data.get_data(format='df')  # 或 'dict'、'DeepPurpose' 等

# 取得訓練/驗證/測試分割
split = data.get_split(method='scaffold', seed=1, frac=[0.7, 0.1, 0.2])
```

## 注意事項

- 資料集大小和統計是近似值，可能會更新
- 新資料集會定期添加到 TDC
- 某些資料集可能需要額外的依賴項
- 請查看官方 TDC 網站以獲取最新的資料集列表：https://tdcommons.ai/overview/
