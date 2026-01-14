# Matchms 過濾函數參考

本文件提供 matchms 中所有可用於處理質譜資料的過濾函數的完整參考。

## 元資料處理過濾器

### 化合物與化學資訊

**add_compound_name(spectrum)**
- 將化合物名稱添加到正確的元資料欄位
- 標準化化合物名稱儲存位置

**clean_compound_name(spectrum)**
- 從化合物名稱中移除常見的不需要附加資訊
- 清理格式不一致

**derive_adduct_from_name(spectrum)**
- 從化合物名稱中擷取加合物資訊
- 將加合物標記移至適當的元資料欄位

**derive_formula_from_name(spectrum)**
- 在化合物名稱中檢測化學式
- 將化學式重新定位到適當的元資料欄位

**derive_annotation_from_compound_name(spectrum)**
- 使用化合物名稱從 PubChem 檢索 SMILES/InChI
- 自動註釋化學結構

### 化學結構轉換

**derive_inchi_from_smiles(spectrum)**
- 從 SMILES 字串生成 InChI
- 需要 rdkit 庫

**derive_inchikey_from_inchi(spectrum)**
- 從 InChI 計算 InChIKey
- 27 字元的雜湊識別碼

**derive_smiles_from_inchi(spectrum)**
- 從 InChI 表示法建立 SMILES
- 需要 rdkit 庫

**repair_inchi_inchikey_smiles(spectrum)**
- 校正放錯位置的化學識別碼
- 修復元資料欄位混淆

**repair_not_matching_annotation(spectrum)**
- 確保 SMILES、InChI 和 InChIKey 之間的一致性
- 驗證化學結構註釋匹配

**add_fingerprint(spectrum, fingerprint_type="daylight", nbits=2048, radius=2)**
- 生成用於相似性計算的分子指紋
- 指紋類型："daylight"、"morgan1"、"morgan2"、"morgan3"
- 與 FingerprintSimilarity 評分一起使用

### 質量與電荷資訊

**add_precursor_mz(spectrum)**
- 正規化前驅離子 m/z 值
- 標準化前驅離子質量元資料

**add_parent_mass(spectrum, estimate_from_adduct=True)**
- 從前驅離子 m/z 和加合物計算中性母離子質量
- 如果無法直接獲得，可從加合物估計

**correct_charge(spectrum)**
- 將電荷值與離子模式對齊
- 確保電荷符號與離子化模式匹配

**make_charge_int(spectrum)**
- 將電荷轉換為整數格式
- 標準化電荷表示

**clean_adduct(spectrum)**
- 標準化加合物標記
- 校正常見的加合物格式問題

**interpret_pepmass(spectrum)**
- 將 pepmass 欄位解析為組成值
- 從組合欄位中擷取前驅離子 m/z 和強度

### 離子模式與驗證

**derive_ionmode(spectrum)**
- 從加合物資訊確定離子模式
- 從加合物類型推斷正/負模式

**require_correct_ionmode(spectrum, ion_mode)**
- 按指定離子模式過濾質譜
- 如果離子模式不匹配則返回 None
- 使用方式：`spectrum = require_correct_ionmode(spectrum, "positive")`

**require_precursor_mz(spectrum, minimum_accepted_mz=0.0)**
- 驗證前驅離子 m/z 的存在和值
- 如果缺失或低於閾值則返回 None

**require_precursor_below_mz(spectrum, maximum_accepted_mz=1000.0)**
- 強制執行前驅離子 m/z 最大限制
- 如果前驅離子超過閾值則返回 None

### 滯留資訊

**add_retention_time(spectrum)**
- 將滯留時間協調為浮點值
- 標準化 RT 元資料欄位

**add_retention_index(spectrum)**
- 將滯留指數儲存在標準化欄位中
- 正規化 RI 元資料

### 資料協調

**harmonize_undefined_inchi(spectrum, undefined="", aliases=None)**
- 標準化未定義/空的 InChI 條目
- 將各種「未知」表示替換為一致的值

**harmonize_undefined_inchikey(spectrum, undefined="", aliases=None)**
- 標準化未定義/空的 InChIKey 條目
- 統一缺失資料的表示

**harmonize_undefined_smiles(spectrum, undefined="", aliases=None)**
- 標準化未定義/空的 SMILES 條目
- 一致處理缺失的結構資料

### 修復與品質函數

**repair_adduct_based_on_smiles(spectrum, mass_tolerance=0.1)**
- 使用 SMILES 和質量匹配校正加合物
- 驗證加合物與計算質量匹配

**repair_parent_mass_is_mol_wt(spectrum, mass_tolerance=0.1)**
- 將分子量轉換為單同位素質量
- 修復常見的元資料混淆

**repair_precursor_is_parent_mass(spectrum)**
- 修復交換的前驅離子/母離子質量值
- 校正欄位錯誤分配

**repair_smiles_of_salts(spectrum, mass_tolerance=0.1)**
- 移除鹽成分以匹配母離子質量
- 擷取相關的分子片段

**require_parent_mass_match_smiles(spectrum, mass_tolerance=0.1)**
- 驗證母離子質量與 SMILES 計算的質量
- 如果質量在容差範圍內不匹配則返回 None

**require_valid_annotation(spectrum)**
- 確保完整、一致的化學註釋
- 驗證 SMILES、InChI 和 InChIKey 的存在和一致性

## 峰值處理過濾器

### 正規化與選擇

**normalize_intensities(spectrum)**
- 將峰值強度縮放至單位高度（最大值 = 1.0）
- 相似性計算的必要預處理步驟

**select_by_intensity(spectrum, intensity_from=0.0, intensity_to=1.0)**
- 保留指定絕對強度範圍內的峰值
- 按原始強度值過濾

**select_by_relative_intensity(spectrum, intensity_from=0.0, intensity_to=1.0)**
- 保持相對強度範圍內的峰值
- 作為最大強度的分數進行過濾

**select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)**
- 按 m/z 值範圍過濾峰值
- 移除指定 m/z 窗口外的峰值

### 峰值縮減與過濾

**reduce_to_number_of_peaks(spectrum, n_max=None, ratio_desired=None)**
- 當超過最大值時移除最低強度的峰值
- 可指定絕對數量或比率
- 使用方式：`spectrum = reduce_to_number_of_peaks(spectrum, n_max=100)`

**remove_peaks_around_precursor_mz(spectrum, mz_tolerance=17)**
- 消除前驅離子容差範圍內的峰值
- 移除前驅離子和同位素峰
- 基於碎片相似性的常見預處理

**remove_peaks_outside_top_k(spectrum, k=10, ratio_desired=None)**
- 只保留接近 k 個最高強度峰值的峰值
- 聚焦於最具資訊性的信號

**require_minimum_number_of_peaks(spectrum, n_required=10)**
- 丟棄峰值不足的質譜
- 品質控制過濾器
- 如果峰值計數低於閾值則返回 None

**require_minimum_number_of_high_peaks(spectrum, n_required=5, intensity_threshold=0.05)**
- 移除缺少高強度峰值的質譜
- 確保資料品質
- 如果超過閾值的峰值不足則返回 None

### 損失計算

**add_losses(spectrum, loss_mz_from=5.0, loss_mz_to=200.0)**
- 從前驅離子質量衍生中性丟失
- 計算損失 = 前驅離子_mz - 碎片_mz
- 為 NeutralLossesCosine 評分添加損失到質譜

## 流程函數

**default_filters(spectrum)**
- 依序應用九個必要的元資料過濾器：
  1. make_charge_int
  2. add_precursor_mz
  3. add_retention_time
  4. add_retention_index
  5. derive_adduct_from_name
  6. derive_formula_from_name
  7. clean_compound_name
  8. harmonize_undefined_smiles
  9. harmonize_undefined_inchi
- 元資料協調的建議起點

**SpectrumProcessor(filters)**
- 協調多過濾器流程
- 接受過濾器函數列表
- 範例：
```python
from matchms import SpectrumProcessor
processor = SpectrumProcessor([
    default_filters,
    normalize_intensities,
    lambda s: select_by_relative_intensity(s, intensity_from=0.01)
])
processed = processor(spectrum)
```

## 常見過濾器組合

### 標準預處理流程
```python
from matchms.filtering import (default_filters, normalize_intensities,
                               select_by_relative_intensity,
                               require_minimum_number_of_peaks)

spectrum = default_filters(spectrum)
spectrum = normalize_intensities(spectrum)
spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
```

### 品質控制流程
```python
from matchms.filtering import (require_precursor_mz, require_minimum_number_of_peaks,
                               require_minimum_number_of_high_peaks)

spectrum = require_precursor_mz(spectrum, minimum_accepted_mz=50.0)
if spectrum is None:
    # 質譜未通過品質控制
    pass
spectrum = require_minimum_number_of_peaks(spectrum, n_required=10)
spectrum = require_minimum_number_of_high_peaks(spectrum, n_required=5)
```

### 化學註釋流程
```python
from matchms.filtering import (derive_inchi_from_smiles, derive_inchikey_from_inchi,
                               add_fingerprint, require_valid_annotation)

spectrum = derive_inchi_from_smiles(spectrum)
spectrum = derive_inchikey_from_inchi(spectrum)
spectrum = add_fingerprint(spectrum, fingerprint_type="morgan2", nbits=2048)
spectrum = require_valid_annotation(spectrum)
```

### 峰值清理流程
```python
from matchms.filtering import (normalize_intensities, remove_peaks_around_precursor_mz,
                               select_by_relative_intensity, reduce_to_number_of_peaks)

spectrum = normalize_intensities(spectrum)
spectrum = remove_peaks_around_precursor_mz(spectrum, mz_tolerance=17)
spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
spectrum = reduce_to_number_of_peaks(spectrum, n_max=200)
```

## 過濾器使用注意事項

1. **順序很重要**：按邏輯順序應用過濾器（例如，在相對強度選擇之前先正規化）
2. **過濾器返回 None**：許多過濾器對無效質譜返回 None；在繼續之前檢查 None
3. **不可變性**：過濾器通常返回修改後的副本；將結果重新分配給變數
4. **流程效率**：使用 SpectrumProcessor 進行一致的多質譜處理
5. **文件**：有關詳細參數，請參閱 matchms.readthedocs.io/en/latest/api/matchms.filtering.html

