# TDC 分子生成預言機

預言機是評估生成分子在特定維度上品質的函數。TDC 提供 17+ 個用於全新藥物設計中分子最佳化任務的預言機函數。

## 概述

預言機測量分子屬性並服務於兩個主要目的：

1. **目標導向生成**：最佳化分子以最大化/最小化特定屬性
2. **分布學習**：評估生成的分子是否符合所需的屬性分布

## 使用預言機

### 基本用法

```python
from tdc import Oracle

# 初始化預言機
oracle = Oracle(name='GSK3B')

# 評估單一分子（SMILES 字串）
score = oracle('CC(C)Cc1ccc(cc1)C(C)C(O)=O')

# 評估多個分子
scores = oracle(['SMILES1', 'SMILES2', 'SMILES3'])
```

### 預言機類別

TDC 預言機根據評估的分子屬性組織為幾個類別。

## 生化預言機

預測對生物標靶的結合親和力或活性。

### 標靶特異性預言機

**DRD2 - 多巴胺受體 D2**
```python
oracle = Oracle(name='DRD2')
score = oracle(smiles)
```
- 測量與 DRD2 受體的結合親和力
- 對神經和精神藥物開發很重要
- 較高分數表示較強的結合

**GSK3B - 糖原合成酶激酶-3 Beta**
```python
oracle = Oracle(name='GSK3B')
score = oracle(smiles)
```
- 預測 GSK3β 抑制
- 與阿茲海默症、糖尿病和癌症研究相關
- 較高分數表示較好的抑制

**JNK3 - c-Jun N-端激酶 3**
```python
oracle = Oracle(name='JNK3')
score = oracle(smiles)
```
- 測量 JNK3 激酶抑制
- 神經退化性疾病的標靶
- 較高分數表示較強的抑制

**5HT2A - 血清素 2A 受體**
```python
oracle = Oracle(name='5HT2A')
score = oracle(smiles)
```
- 預測血清素受體結合
- 對精神藥物很重要
- 較高分數表示較強的結合

**ACE - 血管收縮素轉換酶**
```python
oracle = Oracle(name='ACE')
score = oracle(smiles)
```
- 測量 ACE 抑制
- 高血壓治療的標靶
- 較高分數表示較好的抑制

**MAPK - 絲裂原活化蛋白激酶**
```python
oracle = Oracle(name='MAPK')
score = oracle(smiles)
```
- 預測 MAPK 抑制
- 癌症和發炎疾病的標靶

**CDK - 細胞週期素依賴性激酶**
```python
oracle = Oracle(name='CDK')
score = oracle(smiles)
```
- 測量 CDK 抑制
- 對癌症藥物開發很重要

**P38 - p38 MAP 激酶**
```python
oracle = Oracle(name='P38')
score = oracle(smiles)
```
- 預測 p38 MAPK 抑制
- 發炎疾病的標靶

**PARP1 - 聚(ADP-核糖) 聚合酶 1**
```python
oracle = Oracle(name='PARP1')
score = oracle(smiles)
```
- 測量 PARP1 抑制
- 癌症治療的標靶（DNA 修復機制）

**PIK3CA - 磷脂醯肌醇-4,5-雙磷酸 3-激酶**
```python
oracle = Oracle(name='PIK3CA')
score = oracle(smiles)
```
- 預測 PIK3CA 抑制
- 腫瘤學中的重要標靶

## 物理化學預言機

評估類藥屬性和 ADME 特徵。

### 類藥性預言機

**QED - 類藥性定量估計**
```python
oracle = Oracle(name='QED')
score = oracle(smiles)
```
- 結合多個物理化學屬性
- 分數範圍從 0（非類藥）到 1（類藥）
- 基於 Bickerton 等人的標準

**Lipinski - 五規則**
```python
oracle = Oracle(name='Lipinski')
score = oracle(smiles)
```
- Lipinski 規則違反數量
- 規則：MW ≤ 500、logP ≤ 5、HBD ≤ 5、HBA ≤ 10
- 分數為 0 表示完全符合

### 分子屬性

**SA - 合成可及性**
```python
oracle = Oracle(name='SA')
score = oracle(smiles)
```
- 估計合成的容易程度
- 分數範圍從 1（容易）到 10（困難）
- 較低分數表示更容易合成

**LogP - 辛醇-水分配係數**
```python
oracle = Oracle(name='LogP')
score = oracle(smiles)
```
- 測量親脂性
- 對膜通透性很重要
- 典型類藥範圍：0-5

**MW - 分子量**
```python
oracle = Oracle(name='MW')
score = oracle(smiles)
```
- 回傳分子量（道爾頓）
- 類藥範圍通常為 150-500 Da

## 複合預言機

結合多個屬性進行多目標最佳化。

**Isomer Meta**
```python
oracle = Oracle(name='Isomer_Meta')
score = oracle(smiles)
```
- 評估特定異構體屬性
- 用於立體化學最佳化

**Median Molecules**
```python
oracle = Oracle(name='Median1', 'Median2')
score = oracle(smiles)
```
- 測試生成具有中位數屬性分子的能力
- 對分布學習基準有用

**Rediscovery**
```python
oracle = Oracle(name='Rediscovery')
score = oracle(smiles)
```
- 測量與已知參考分子的相似度
- 測試重新生成現有藥物的能力

**Similarity**
```python
oracle = Oracle(name='Similarity')
score = oracle(smiles)
```
- 計算與目標分子的結構相似度
- 基於分子指紋（通常是 Tanimoto 相似度）

**Uniqueness**
```python
oracle = Oracle(name='Uniqueness')
scores = oracle(smiles_list)
```
- 測量生成分子集中的多樣性
- 回傳唯一分子的比例

**Novelty**
```python
oracle = Oracle(name='Novelty')
scores = oracle(smiles_list, training_set)
```
- 測量生成分子與訓練集的差異程度
- 較高分數表示更新穎的結構

## 專業預言機

**ASKCOS - 逆合成評分**
```python
oracle = Oracle(name='ASKCOS')
score = oracle(smiles)
```
- 使用逆合成評估合成可行性
- 需要 ASKCOS 後端（IBM RXN）
- 基於逆合成路徑可用性評分

**Docking Score**
```python
oracle = Oracle(name='Docking')
score = oracle(smiles)
```
- 對目標蛋白質的分子對接分數
- 需要蛋白質結構和對接軟體
- 較低分數通常表示較好的結合

**Vina - AutoDock Vina 分數**
```python
oracle = Oracle(name='Vina')
score = oracle(smiles)
```
- 使用 AutoDock Vina 進行蛋白質-配體對接
- 預測結合親和力（kcal/mol）
- 更負的分數表示更強的結合

## 多目標最佳化

結合多個預言機進行多屬性最佳化：

```python
from tdc import Oracle

# 初始化多個預言機
qed_oracle = Oracle(name='QED')
sa_oracle = Oracle(name='SA')
drd2_oracle = Oracle(name='DRD2')

# 定義自訂評分函數
def multi_objective_score(smiles):
    qed = qed_oracle(smiles)
    sa = 1 / (1 + sa_oracle(smiles))  # 反轉 SA（較低較好）
    drd2 = drd2_oracle(smiles)

    # 加權組合
    return 0.3 * qed + 0.3 * sa + 0.4 * drd2

# 評估分子
score = multi_objective_score('CC(C)Cc1ccc(cc1)C(C)C(O)=O')
```

## 預言機效能考量

### 速度
- **快速**：QED、SA、LogP、MW、Lipinski（基於規則的計算）
- **中等**：標靶特異性機器學習模型（DRD2、GSK3B 等）
- **慢速**：基於對接的預言機（Vina、ASKCOS）

### 可靠性
- 預言機是在特定資料集上訓練的機器學習模型
- 可能無法泛化到所有化學空間
- 使用多個預言機驗證結果

### 批次處理
```python
# 高效的批次評估
oracle = Oracle(name='GSK3B')
smiles_list = ['SMILES1', 'SMILES2', ..., 'SMILES1000']
scores = oracle(smiles_list)  # 比個別呼叫更快
```

## 常見工作流程

### 目標導向生成
```python
from tdc import Oracle
from tdc.generation import MolGen

# 載入訓練資料
data = MolGen(name='ChEMBL_V29')
train_smiles = data.get_data()['Drug'].tolist()

# 初始化預言機
oracle = Oracle(name='GSK3B')

# 生成分子（使用者實作生成模型）
# generated_smiles = generator.generate(n=1000)

# 評估生成的分子
scores = oracle(generated_smiles)
best_molecules = [(s, score) for s, score in zip(generated_smiles, scores)]
best_molecules.sort(key=lambda x: x[1], reverse=True)

print(f"Top 10 molecules:")
for smiles, score in best_molecules[:10]:
    print(f"{smiles}: {score:.3f}")
```

### 分布學習
```python
from tdc import Oracle
import numpy as np

# 初始化預言機
oracle = Oracle(name='QED')

# 評估訓練集
train_scores = oracle(train_smiles)
train_mean = np.mean(train_scores)
train_std = np.std(train_scores)

# 評估生成集
gen_scores = oracle(generated_smiles)
gen_mean = np.mean(gen_scores)
gen_std = np.std(gen_scores)

# 比較分布
print(f"Training: μ={train_mean:.3f}, σ={train_std:.3f}")
print(f"Generated: μ={gen_mean:.3f}, σ={gen_std:.3f}")
```

## 與 TDC 基準整合

```python
from tdc.generation import MolGen

# 與 GuacaMol 基準一起使用
data = MolGen(name='GuacaMol')

# 預言機自動整合
# 每個 GuacaMol 任務都有關聯的預言機
benchmark_results = data.evaluate_guacamol(
    generated_molecules=your_molecules,
    oracle_name='GSK3B'
)
```

## 注意事項

- 預言機分數是預測值，不是實驗測量值
- 始終透過實驗驗證頂級候選者
- 不同的預言機可能有不同的分數範圍和解釋
- 某些預言機需要額外的依賴項或 API 存取
- 查看預言機文件以獲取具體詳情：https://tdcommons.ai/functions/oracles/

## 添加自訂預言機

要建立自訂預言機函數：

```python
class CustomOracle:
    def __init__(self):
        # 初始化您的模型/方法
        pass

    def __call__(self, smiles):
        # 實作您的評分邏輯
        # 回傳分數或分數列表
        pass

# 像內建預言機一樣使用
custom_oracle = CustomOracle()
score = custom_oracle('CC(C)Cc1ccc(cc1)C(C)C(O)=O')
```

## 參考文獻

- TDC 預言機文件：https://tdcommons.ai/functions/oracles/
- GuacaMol 論文："GuacaMol: Benchmarking Models for de Novo Molecular Design"
- MOSES 論文："Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models"
