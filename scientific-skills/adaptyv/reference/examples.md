# 程式碼範例

## 設定與身份驗證

### 基本設定

```python
import os
import requests
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 配置
API_KEY = os.getenv("ADAPTYV_API_KEY")
BASE_URL = "https://kq5jp7qj7wdqklhsxmovkzn4l40obksv.lambda-url.eu-central-1.on.aws"

# 標準標頭
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def check_api_connection():
    """驗證 API 連線和憑證"""
    try:
        response = requests.get(f"{BASE_URL}/organization/credits", headers=HEADERS)
        response.raise_for_status()
        print("✓ API 連線成功")
        print(f"  剩餘點數：{response.json()['balance']}")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"✗ API 身份驗證失敗：{e}")
        return False
```

### 環境設定

建立 `.env` 檔案：
```bash
ADAPTYV_API_KEY=your_api_key_here
```

安裝依賴套件：
```bash
uv pip install requests python-dotenv
```

## 實驗提交

### 提交單一序列

```python
def submit_single_experiment(sequence, experiment_type="binding", target_id=None):
    """
    提交單一蛋白質序列進行測試

    參數：
        sequence：胺基酸序列字串
        experiment_type：實驗類型（binding、expression、thermostability、enzyme_activity）
        target_id：可選的結合檢測標靶識別碼

    回傳：
        實驗 ID 和狀態
    """

    # 格式化為 FASTA
    fasta_content = f">protein_sequence\n{sequence}\n"

    payload = {
        "sequences": fasta_content,
        "experiment_type": experiment_type
    }

    if target_id:
        payload["target_id"] = target_id

    response = requests.post(
        f"{BASE_URL}/experiments",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()
    result = response.json()

    print(f"✓ 實驗已提交")
    print(f"  實驗 ID：{result['experiment_id']}")
    print(f"  狀態：{result['status']}")
    print(f"  預計完成時間：{result['estimated_completion']}")

    return result

# 使用範例
sequence = "MKVLWAALLGLLGAAAAFPAVTSAVKPYKAAVSAAVSKPYKAAVSAAVSKPYK"
experiment = submit_single_experiment(sequence, experiment_type="expression")
```

### 提交多個序列（批次）

```python
def submit_batch_experiment(sequences_dict, experiment_type="binding", metadata=None):
    """
    在單一批次中提交多個蛋白質序列

    參數：
        sequences_dict：{name: sequence} 的字典
        experiment_type：實驗類型
        metadata：可選的附加資訊字典

    回傳：
        實驗詳情
    """

    # 將所有序列格式化為 FASTA
    fasta_content = ""
    for name, sequence in sequences_dict.items():
        fasta_content += f">{name}\n{sequence}\n"

    payload = {
        "sequences": fasta_content,
        "experiment_type": experiment_type
    }

    if metadata:
        payload["metadata"] = metadata

    response = requests.post(
        f"{BASE_URL}/experiments",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()
    result = response.json()

    print(f"✓ 批次實驗已提交")
    print(f"  實驗 ID：{result['experiment_id']}")
    print(f"  序列數：{len(sequences_dict)}")
    print(f"  狀態：{result['status']}")

    return result

# 使用範例
sequences = {
    "variant_1": "MKVLWAALLGLLGAAA...",
    "variant_2": "MKVLSAALLGLLGAAA...",
    "variant_3": "MKVLAAALLGLLGAAA...",
    "wildtype": "MKVLWAALLGLLGAAA..."
}

metadata = {
    "project": "antibody_optimization",
    "round": 3,
    "notes": "Testing solubility-optimized variants"
}

experiment = submit_batch_experiment(sequences, "expression", metadata)
```

### 提交並設定 Webhook 通知

```python
def submit_with_webhook(sequences_dict, experiment_type, webhook_url):
    """
    提交實驗並設定完成通知的 webhook

    參數：
        sequences_dict：{name: sequence} 的字典
        experiment_type：實驗類型
        webhook_url：接收完成通知的 URL
    """

    fasta_content = ""
    for name, sequence in sequences_dict.items():
        fasta_content += f">{name}\n{sequence}\n"

    payload = {
        "sequences": fasta_content,
        "experiment_type": experiment_type,
        "webhook_url": webhook_url
    }

    response = requests.post(
        f"{BASE_URL}/experiments",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()
    result = response.json()

    print(f"✓ 實驗已提交並設定 webhook")
    print(f"  實驗 ID：{result['experiment_id']}")
    print(f"  Webhook：{webhook_url}")

    return result

# 範例
webhook_url = "https://your-server.com/adaptyv-webhook"
experiment = submit_with_webhook(sequences, "binding", webhook_url)
```

## 追蹤實驗

### 檢查實驗狀態

```python
def check_experiment_status(experiment_id):
    """
    取得實驗的目前狀態

    參數：
        experiment_id：實驗識別碼

    回傳：
        狀態資訊
    """

    response = requests.get(
        f"{BASE_URL}/experiments/{experiment_id}",
        headers=HEADERS
    )

    response.raise_for_status()
    status = response.json()

    print(f"實驗：{experiment_id}")
    print(f"  狀態：{status['status']}")
    print(f"  建立時間：{status['created_at']}")
    print(f"  更新時間：{status['updated_at']}")

    if 'progress' in status:
        print(f"  進度：{status['progress']['percentage']}%")
        print(f"  目前階段：{status['progress']['stage']}")

    return status

# 範例
status = check_experiment_status("exp_abc123xyz")
```

### 列出所有實驗

```python
def list_experiments(status_filter=None, limit=50):
    """
    列出實驗並可選擇按狀態篩選

    參數：
        status_filter：按狀態篩選（submitted、processing、completed、failed）
        limit：最大結果數

    回傳：
        實驗列表
    """

    params = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    response = requests.get(
        f"{BASE_URL}/experiments",
        headers=HEADERS,
        params=params
    )

    response.raise_for_status()
    result = response.json()

    print(f"找到 {result['total']} 個實驗")
    for exp in result['experiments']:
        print(f"  {exp['experiment_id']}：{exp['status']}（{exp['experiment_type']}）")

    return result['experiments']

# 範例 - 列出所有已完成的實驗
completed_experiments = list_experiments(status_filter="completed")
```

### 輪詢直到完成

```python
import time

def wait_for_completion(experiment_id, check_interval=3600):
    """
    輪詢實驗狀態直到完成

    參數：
        experiment_id：實驗識別碼
        check_interval：狀態檢查間隔秒數（預設：1 小時）

    回傳：
        最終狀態
    """

    print(f"正在監控實驗 {experiment_id}...")

    while True:
        status = check_experiment_status(experiment_id)

        if status['status'] == 'completed':
            print("✓ 實驗已完成！")
            return status
        elif status['status'] == 'failed':
            print("✗ 實驗失敗")
            return status

        print(f"  狀態：{status['status']} - 將在 {check_interval} 秒後再次檢查")
        time.sleep(check_interval)

# 範例（不建議 - 請改用 webhooks！）
# status = wait_for_completion("exp_abc123xyz", check_interval=3600)
```

## 擷取結果

### 下載實驗結果

```python
import json

def download_results(experiment_id, output_dir="results"):
    """
    下載並解析實驗結果

    參數：
        experiment_id：實驗識別碼
        output_dir：儲存結果的目錄

    回傳：
        解析後的結果資料
    """

    # 取得結果
    response = requests.get(
        f"{BASE_URL}/experiments/{experiment_id}/results",
        headers=HEADERS
    )

    response.raise_for_status()
    results = response.json()

    # 儲存結果 JSON
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{experiment_id}_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ 結果已下載：{output_file}")
    print(f"  測試的序列數：{len(results['results'])}")

    # 如果有原始資料可供下載
    if 'download_urls' in results:
        for data_type, url in results['download_urls'].items():
            print(f"  {data_type} 可在此取得：{url}")

    return results

# 範例
results = download_results("exp_abc123xyz")
```

### 解析結合結果

```python
import pandas as pd

def parse_binding_results(results):
    """
    將結合檢測結果解析為 DataFrame

    參數：
        results：來自 API 的結果字典

    回傳：
        包含整理後結果的 pandas DataFrame
    """

    data = []
    for result in results['results']:
        row = {
            'sequence_id': result['sequence_id'],
            'kd': result['measurements']['kd'],
            'kd_error': result['measurements']['kd_error'],
            'kon': result['measurements']['kon'],
            'koff': result['measurements']['koff'],
            'confidence': result['quality_metrics']['confidence'],
            'r_squared': result['quality_metrics']['r_squared']
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 按親和力排序（較低的 KD = 較強的結合）
    df = df.sort_values('kd')

    print("前 5 名結合者：")
    print(df.head())

    return df

# 範例
experiment_id = "exp_abc123xyz"
results = download_results(experiment_id)
binding_df = parse_binding_results(results)

# 匯出為 CSV
binding_df.to_csv(f"{experiment_id}_binding_results.csv", index=False)
```

### 解析表現結果

```python
def parse_expression_results(results):
    """
    將表現測試結果解析為 DataFrame

    參數：
        results：來自 API 的結果字典

    回傳：
        包含整理後結果的 pandas DataFrame
    """

    data = []
    for result in results['results']:
        row = {
            'sequence_id': result['sequence_id'],
            'yield_mg_per_l': result['measurements']['total_yield_mg_per_l'],
            'soluble_fraction': result['measurements']['soluble_fraction_percent'],
            'purity': result['measurements']['purity_percent'],
            'percentile': result['ranking']['percentile']
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 按產量排序
    df = df.sort_values('yield_mg_per_l', ascending=False)

    print(f"平均產量：{df['yield_mg_per_l'].mean():.2f} mg/L")
    print(f"表現最佳者：{df.iloc[0]['sequence_id']}（{df.iloc[0]['yield_mg_per_l']:.2f} mg/L）")

    return df

# 範例
results = download_results("exp_expression123")
expression_df = parse_expression_results(results)
```

## 標靶目錄

### 搜尋標靶

```python
def search_targets(query, species=None, category=None):
    """
    搜尋抗原目錄

    參數：
        query：搜尋詞（蛋白質名稱、UniProt ID 等）
        species：可選的物種篩選
        category：可選的類別篩選

    回傳：
        符合的標靶列表
    """

    params = {"search": query}
    if species:
        params["species"] = species
    if category:
        params["category"] = category

    response = requests.get(
        f"{BASE_URL}/targets",
        headers=HEADERS,
        params=params
    )

    response.raise_for_status()
    targets = response.json()['targets']

    print(f"找到 {len(targets)} 個符合「{query}」的標靶：")
    for target in targets:
        print(f"  {target['target_id']}：{target['name']}")
        print(f"    物種：{target['species']}")
        print(f"    可用性：{target['availability']}")
        print(f"    價格：${target['price_usd']}")

    return targets

# 範例
targets = search_targets("PD-L1", species="Homo sapiens")
```

### 請求自訂標靶

```python
def request_custom_target(target_name, uniprot_id=None, species=None, notes=None):
    """
    請求標準目錄中沒有的自訂抗原

    參數：
        target_name：標靶蛋白質名稱
        uniprot_id：可選的 UniProt 識別碼
        species：物種名稱
        notes：附加需求或備註

    回傳：
        請求確認
    """

    payload = {
        "target_name": target_name,
        "species": species
    }

    if uniprot_id:
        payload["uniprot_id"] = uniprot_id
    if notes:
        payload["notes"] = notes

    response = requests.post(
        f"{BASE_URL}/targets/request",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()
    result = response.json()

    print(f"✓ 自訂標靶請求已提交")
    print(f"  請求 ID：{result['request_id']}")
    print(f"  狀態：{result['status']}")

    return result

# 範例
request = request_custom_target(
    target_name="Novel receptor XYZ",
    uniprot_id="P12345",
    species="Mus musculus",
    notes="Need high purity for structural studies"
)
```

## 完整工作流程

### 端對端結合檢測

```python
def complete_binding_workflow(sequences_dict, target_id, project_name):
    """
    完整工作流程：提交序列、追蹤並擷取結合結果

    參數：
        sequences_dict：{name: sequence} 的字典
        target_id：來自目錄的標靶識別碼
        project_name：metadata 的專案名稱

    回傳：
        包含結合結果的 DataFrame
    """

    print("=== 開始結合檢測工作流程 ===")

    # 步驟 1：提交實驗
    print("\n1. 提交實驗...")
    metadata = {
        "project": project_name,
        "target": target_id
    }

    experiment = submit_batch_experiment(
        sequences_dict,
        experiment_type="binding",
        metadata=metadata
    )

    experiment_id = experiment['experiment_id']

    # 步驟 2：儲存實驗資訊
    print("\n2. 儲存實驗詳情...")
    with open(f"{experiment_id}_info.json", 'w') as f:
        json.dump(experiment, f, indent=2)

    print(f"✓ 實驗 {experiment_id} 已提交")
    print("  結果將在約 21 天後可用")
    print("  使用 webhook 或輪詢狀態以獲取更新")

    # 注意：實際上，在此步驟之前需等待完成
    # print("\n3. 等待完成...")
    # status = wait_for_completion(experiment_id)

    # print("\n4. 下載結果...")
    # results = download_results(experiment_id)

    # print("\n5. 解析結果...")
    # df = parse_binding_results(results)

    # return df

    return experiment_id

# 範例
antibody_variants = {
    "variant_1": "EVQLVESGGGLVQPGG...",
    "variant_2": "EVQLVESGGGLVQPGS...",
    "variant_3": "EVQLVESGGGLVQPGA...",
    "wildtype": "EVQLVESGGGLVQPGG..."
}

experiment_id = complete_binding_workflow(
    antibody_variants,
    target_id="tgt_pdl1_human",
    project_name="antibody_affinity_maturation"
)
```

### 優化 + 測試流程

```python
# 結合計算優化與實驗測試

def optimization_and_testing_pipeline(initial_sequences, experiment_type="expression"):
    """
    完整流程：計算優化序列，然後提交測試

    參數：
        initial_sequences：{name: sequence} 的字典
        experiment_type：實驗類型

    回傳：
        用於追蹤的實驗 ID
    """

    print("=== 優化與測試流程 ===")

    # 步驟 1：計算優化
    print("\n1. 計算優化...")
    from protein_optimization import complete_optimization_pipeline

    optimized = complete_optimization_pipeline(initial_sequences)

    print(f"✓ 優化完成")
    print(f"  起始序列數：{len(initial_sequences)}")
    print(f"  優化後序列數：{len(optimized)}")

    # 步驟 2：選擇最佳候選者
    print("\n2. 選擇最佳候選者進行測試...")
    top_candidates = optimized[:50]  # 前 50 名

    sequences_to_test = {
        seq_data['name']: seq_data['sequence']
        for seq_data in top_candidates
    }

    # 步驟 3：提交實驗驗證
    print("\n3. 提交至 Adaptyv...")
    metadata = {
        "optimization_method": "computational_pipeline",
        "initial_library_size": len(initial_sequences),
        "computational_scores": [s['combined'] for s in top_candidates]
    }

    experiment = submit_batch_experiment(
        sequences_to_test,
        experiment_type=experiment_type,
        metadata=metadata
    )

    print(f"✓ 流程完成")
    print(f"  實驗 ID：{experiment['experiment_id']}")

    return experiment['experiment_id']

# 範例
initial_library = {
    f"variant_{i}": generate_random_sequence()
    for i in range(1000)
}

experiment_id = optimization_and_testing_pipeline(
    initial_library,
    experiment_type="expression"
)
```

### 批次結果分析

```python
def analyze_multiple_experiments(experiment_ids):
    """
    下載並分析多個實驗的結果

    參數：
        experiment_ids：實驗識別碼列表

    回傳：
        包含所有結果的合併 DataFrame
    """

    all_results = []

    for exp_id in experiment_ids:
        print(f"處理 {exp_id}...")

        # 下載結果
        results = download_results(exp_id, output_dir=f"results/{exp_id}")

        # 根據實驗類型解析
        exp_type = results.get('experiment_type', 'unknown')

        if exp_type == 'binding':
            df = parse_binding_results(results)
            df['experiment_id'] = exp_id
            all_results.append(df)

        elif exp_type == 'expression':
            df = parse_expression_results(results)
            df['experiment_id'] = exp_id
            all_results.append(df)

    # 合併所有結果
    combined_df = pd.concat(all_results, ignore_index=True)

    print(f"\n✓ 分析完成")
    print(f"  總實驗數：{len(experiment_ids)}")
    print(f"  總序列數：{len(combined_df)}")

    return combined_df

# 範例
experiment_ids = [
    "exp_round1_abc",
    "exp_round2_def",
    "exp_round3_ghi"
]

all_data = analyze_multiple_experiments(experiment_ids)
all_data.to_csv("combined_results.csv", index=False)
```

## 錯誤處理

### 健壯的 API 包裝器

```python
import time
from requests.exceptions import RequestException, HTTPError

def api_request_with_retry(method, url, max_retries=3, backoff_factor=2, **kwargs):
    """
    帶有重試邏輯和錯誤處理的 API 請求

    參數：
        method：HTTP 方法（GET、POST 等）
        url：請求 URL
        max_retries：最大重試次數
        backoff_factor：指數退避乘數
        **kwargs：傳遞給 requests 的附加參數

    回傳：
        Response 物件

    例外：
        RequestException：如果所有重試都失敗
    """

    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        except HTTPError as e:
            if e.response.status_code == 429:  # 速率限制
                wait_time = backoff_factor ** attempt
                print(f"已達速率限制。等待 {wait_time} 秒...")
                time.sleep(wait_time)
                continue

            elif e.response.status_code >= 500:  # 伺服器錯誤
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"伺服器錯誤。將在 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

            else:  # 客戶端錯誤（4xx）- 不重試
                error_data = e.response.json() if e.response.content else {}
                print(f"API 錯誤：{error_data.get('error', {}).get('message', str(e))}")
                raise

        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"請求失敗。將在 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                continue
            else:
                raise

    raise RequestException(f"在 {max_retries} 次嘗試後失敗")

# 使用範例
response = api_request_with_retry(
    "POST",
    f"{BASE_URL}/experiments",
    headers=HEADERS,
    json={"sequences": fasta_content, "experiment_type": "binding"}
)
```

## 工具函數

### 驗證 FASTA 格式

```python
def validate_fasta(fasta_string):
    """
    驗證 FASTA 格式和序列

    參數：
        fasta_string：FASTA 格式的字串

    回傳：
        (is_valid, error_message) 的元組
    """

    lines = fasta_string.strip().split('\n')

    if not lines:
        return False, "FASTA 內容為空"

    if not lines[0].startswith('>'):
        return False, "FASTA 必須以標頭行（>）開始"

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    current_header = None

    for i, line in enumerate(lines):
        if line.startswith('>'):
            if not line[1:].strip():
                return False, f"第 {i+1} 行：空標頭"
            current_header = line[1:].strip()

        else:
            if current_header is None:
                return False, f"第 {i+1} 行：標頭前出現序列"

            sequence = line.strip().upper()
            invalid = set(sequence) - valid_amino_acids

            if invalid:
                return False, f"第 {i+1} 行：無效的胺基酸：{invalid}"

    return True, None

# 範例
fasta = ">protein1\nMKVLWAALLG\n>protein2\nMATGVLWALG"
is_valid, error = validate_fasta(fasta)

if is_valid:
    print("✓ FASTA 格式有效")
else:
    print(f"✗ FASTA 驗證失敗：{error}")
```

### 將序列格式化為 FASTA

```python
def sequences_to_fasta(sequences_dict):
    """
    將序列字典轉換為 FASTA 格式

    參數：
        sequences_dict：{name: sequence} 的字典

    回傳：
        FASTA 格式的字串
    """

    fasta_content = ""
    for name, sequence in sequences_dict.items():
        # 清理序列（移除空白、確保大寫）
        clean_seq = ''.join(sequence.split()).upper()

        # 驗證
        is_valid, error = validate_fasta(f">{name}\n{clean_seq}")
        if not is_valid:
            raise ValueError(f"無效的序列「{name}」：{error}")

        fasta_content += f">{name}\n{clean_seq}\n"

    return fasta_content

# 範例
sequences = {
    "var1": "MKVLWAALLG",
    "var2": "MATGVLWALG"
}

fasta = sequences_to_fasta(sequences)
print(fasta)
```
