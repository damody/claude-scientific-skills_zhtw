# UniProt API 範例

與 UniProt REST API 互動的多種程式語言實用程式碼範例。

## Python 範例

### 範例 1：基本搜尋
```python
import requests

# 搜尋人類胰島素蛋白質
url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": "insulin AND organism_id:9606 AND reviewed:true",
    "format": "json",
    "size": 10
}

response = requests.get(url, params=params)
data = response.json()

for result in data['results']:
    print(f"{result['primaryAccession']}: {result['proteinDescription']['recommendedName']['fullName']['value']}")
```

### 範例 2：檢索蛋白質序列
```python
import requests

# 以 FASTA 格式取得人類胰島素序列
accession = "P01308"
url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

response = requests.get(url)
print(response.text)
```

### 範例 3：自訂欄位
```python
import requests

# 只取得特定欄位
url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": "gene:BRCA1 AND reviewed:true",
    "format": "tsv",
    "fields": "accession,gene_names,organism_name,length,cc_function"
}

response = requests.get(url, params=params)
print(response.text)
```

### 範例 4：ID 映射
```python
import requests
import time

def map_uniprot_ids(ids, from_db, to_db):
    # 提交作業
    submit_url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": from_db,
        "to": to_db,
        "ids": ",".join(ids)
    }

    response = requests.post(submit_url, data=data)
    job_id = response.json()["jobId"]

    # 輪詢完成狀態
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        response = requests.get(status_url)
        status = response.json()
        if "results" in status or "failedIds" in status:
            break
        time.sleep(3)

    # 取得結果
    results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
    response = requests.get(results_url)
    return response.json()

# 將 UniProt ID 映射到 PDB
ids = ["P01308", "P04637"]
mapping = map_uniprot_ids(ids, "UniProtKB_AC-ID", "PDB")
print(mapping)
```

### 範例 5：串流大型結果
```python
import requests

# 串流所有已審核的人類蛋白質
url = "https://rest.uniprot.org/uniprotkb/stream"
params = {
    "query": "organism_id:9606 AND reviewed:true",
    "format": "fasta"
}

response = requests.get(url, params=params, stream=True)

# 分塊處理
with open("human_proteins.fasta", "w") as f:
    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
        if chunk:
            f.write(chunk)
```

### 範例 6：分頁
```python
import requests

def get_all_results(query, fields=None):
    """取得所有分頁結果"""
    url = "https://rest.uniprot.org/uniprotkb/search"
    all_results = []

    params = {
        "query": query,
        "format": "json",
        "size": 500  # 每頁最大筆數
    }

    if fields:
        params["fields"] = ",".join(fields)

    while True:
        response = requests.get(url, params=params)
        data = response.json()
        all_results.extend(data['results'])

        # 檢查是否有下一頁
        if 'next' in data:
            url = data['next']
        else:
            break

    return all_results

# 取得所有人類激酶
results = get_all_results(
    "protein_name:kinase AND organism_id:9606 AND reviewed:true",
    fields=["accession", "gene_names", "protein_name"]
)
print(f"找到 {len(results)} 個蛋白質")
```

## cURL 範例

### 範例 1：簡單搜尋
```bash
# 搜尋胰島素蛋白質
curl "https://rest.uniprot.org/uniprotkb/search?query=insulin&format=json&size=5"
```

### 範例 2：取得蛋白質條目
```bash
# 以 FASTA 格式取得人類胰島素
curl "https://rest.uniprot.org/uniprotkb/P01308.fasta"
```

### 範例 3：自訂欄位
```bash
# 以 TSV 格式取得特定欄位
curl "https://rest.uniprot.org/uniprotkb/search?query=gene:BRCA1&format=tsv&fields=accession,gene_names,length"
```

### 範例 4：ID 映射 - 提交作業
```bash
# 提交映射作業
curl -X POST "https://rest.uniprot.org/idmapping/run" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "from=UniProtKB_AC-ID&to=PDB&ids=P01308,P04637"
```

### 範例 5：ID 映射 - 取得結果
```bash
# 取得映射結果（替換 JOB_ID）
curl "https://rest.uniprot.org/idmapping/results/JOB_ID"
```

### 範例 6：下載所有結果
```bash
# 下載所有人類已審核蛋白質
curl "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&format=fasta" \
  -o human_proteins.fasta
```

## R 範例

### 範例 1：基本搜尋
```r
library(httr)
library(jsonlite)

# 搜尋胰島素蛋白質
url <- "https://rest.uniprot.org/uniprotkb/search"
query_params <- list(
  query = "insulin AND organism_id:9606",
  format = "json",
  size = 10
)

response <- GET(url, query = query_params)
data <- fromJSON(content(response, "text"))

# 提取登錄號和名稱
proteins <- data$results[, c("primaryAccession", "proteinDescription")]
print(proteins)
```

### 範例 2：取得序列
```r
library(httr)

# 取得蛋白質序列
accession <- "P01308"
url <- paste0("https://rest.uniprot.org/uniprotkb/", accession, ".fasta")

response <- GET(url)
sequence <- content(response, "text")
cat(sequence)
```

### 範例 3：下載到資料框
```r
library(httr)
library(readr)

# 以 TSV 格式取得資料
url <- "https://rest.uniprot.org/uniprotkb/search"
query_params <- list(
  query = "gene:BRCA1 AND reviewed:true",
  format = "tsv",
  fields = "accession,gene_names,organism_name,length"
)

response <- GET(url, query = query_params)
data <- read_tsv(content(response, "text"))
print(data)
```

## JavaScript 範例

### 範例 1：Fetch API
```javascript
// 搜尋蛋白質
async function searchUniProt(query) {
  const url = `https://rest.uniprot.org/uniprotkb/search?query=${encodeURIComponent(query)}&format=json&size=10`;

  const response = await fetch(url);
  const data = await response.json();

  return data.results;
}

// 使用方式
searchUniProt("insulin AND organism_id:9606")
  .then(results => console.log(results));
```

### 範例 2：取得蛋白質條目
```javascript
async function getProtein(accession, format = "json") {
  const url = `https://rest.uniprot.org/uniprotkb/${accession}.${format}`;

  const response = await fetch(url);

  if (format === "json") {
    return await response.json();
  } else {
    return await response.text();
  }
}

// 使用方式
getProtein("P01308", "fasta")
  .then(sequence => console.log(sequence));
```

### 範例 3：ID 映射
```javascript
async function mapIds(ids, fromDb, toDb) {
  // 提交作業
  const submitUrl = "https://rest.uniprot.org/idmapping/run";
  const formData = new URLSearchParams({
    from: fromDb,
    to: toDb,
    ids: ids.join(",")
  });

  const submitResponse = await fetch(submitUrl, {
    method: "POST",
    body: formData
  });
  const { jobId } = await submitResponse.json();

  // 輪詢完成狀態
  const statusUrl = `https://rest.uniprot.org/idmapping/status/${jobId}`;
  while (true) {
    const statusResponse = await fetch(statusUrl);
    const status = await statusResponse.json();

    if ("results" in status || "failedIds" in status) {
      break;
    }

    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  // 取得結果
  const resultsUrl = `https://rest.uniprot.org/idmapping/results/${jobId}`;
  const resultsResponse = await fetch(resultsUrl);
  return await resultsResponse.json();
}

// 使用方式
mapIds(["P01308", "P04637"], "UniProtKB_AC-ID", "PDB")
  .then(mapping => console.log(mapping));
```

## 進階範例

### 範例：具有速率限制的批次處理
```python
import requests
import time
from typing import List, Dict

class UniProtClient:
    def __init__(self, rate_limit=1.0):
        self.base_url = "https://rest.uniprot.org"
        self.rate_limit = rate_limit
        self.last_request = 0

    def _rate_limit(self):
        """強制執行速率限制"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def batch_get_proteins(self, accessions: List[str],
                          batch_size: int = 100) -> List[Dict]:
        """批次取得蛋白質"""
        results = []

        for i in range(0, len(accessions), batch_size):
            batch = accessions[i:i + batch_size]
            query = " OR ".join([f"accession:{acc}" for acc in batch])

            self._rate_limit()

            response = requests.get(
                f"{self.base_url}/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": batch_size
                }
            )

            if response.ok:
                data = response.json()
                results.extend(data.get('results', []))
            else:
                print(f"批次 {i//batch_size} 發生錯誤：{response.status_code}")

        return results

# 使用方式
client = UniProtClient(rate_limit=0.5)
accessions = ["P01308", "P04637", "P12345", "Q9Y6K9"]
proteins = client.batch_get_proteins(accessions)
```

### 範例：具有進度條的下載
```python
import requests
from tqdm import tqdm

def download_with_progress(query, output_file, format="fasta"):
    """具有進度條的結果下載"""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": format
    }

    response = requests.get(url, params=params, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_file, 'wb') as f, \
         tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

# 使用方式
download_with_progress(
    "organism_id:9606 AND reviewed:true",
    "human_proteome.fasta"
)
```

## 資源

- API 文件：https://www.uniprot.org/help/api
- 互動式 API 探索器：https://www.uniprot.org/api-documentation
- Python 客戶端（Unipressed）：https://github.com/multimeric/Unipressed
- Bioservices 套件：https://bioservices.readthedocs.io/
