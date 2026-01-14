# NetworkX 輸入/輸出

## 從檔案讀取圖

### 鄰接列表格式
```python
# 讀取鄰接列表（簡單文字格式）
G = nx.read_adjlist('graph.adjlist')

# 帶節點類型轉換
G = nx.read_adjlist('graph.adjlist', nodetype=int)

# 對於有向圖
G = nx.read_adjlist('graph.adjlist', create_using=nx.DiGraph())

# 寫入鄰接列表
nx.write_adjlist(G, 'graph.adjlist')
```

鄰接列表格式範例：
```
# 節點 鄰居
0 1 2
1 0 3 4
2 0 3
3 1 2 4
4 1 3
```

### 邊列表格式
```python
# 讀取邊列表
G = nx.read_edgelist('graph.edgelist')

# 帶節點類型和邊資料
G = nx.read_edgelist('graph.edgelist',
                     nodetype=int,
                     data=(('weight', float),))

# 讀取加權邊列表
G = nx.read_weighted_edgelist('weighted.edgelist')

# 寫入邊列表
nx.write_edgelist(G, 'graph.edgelist')

# 寫入加權邊列表
nx.write_weighted_edgelist(G, 'weighted.edgelist')
```

邊列表格式範例：
```
# 來源 目標
0 1
1 2
2 3
3 0
```

加權邊列表範例：
```
# 來源 目標 權重
0 1 0.5
1 2 1.0
2 3 0.75
```

### GML（圖建模語言）
```python
# 讀取 GML（保留所有屬性）
G = nx.read_gml('graph.gml')

# 寫入 GML
nx.write_gml(G, 'graph.gml')
```

### GraphML 格式
```python
# 讀取 GraphML（基於 XML 的格式）
G = nx.read_graphml('graph.graphml')

# 寫入 GraphML
nx.write_graphml(G, 'graph.graphml')

# 指定編碼
nx.write_graphml(G, 'graph.graphml', encoding='utf-8')
```

### GEXF（圖交換 XML 格式）
```python
# 讀取 GEXF
G = nx.read_gexf('graph.gexf')

# 寫入 GEXF
nx.write_gexf(G, 'graph.gexf')
```

### Pajek 格式
```python
# 讀取 Pajek .net 檔案
G = nx.read_pajek('graph.net')

# 寫入 Pajek 格式
nx.write_pajek(G, 'graph.net')
```

### LEDA 格式
```python
# 讀取 LEDA 格式
G = nx.read_leda('graph.leda')

# 寫入 LEDA 格式
nx.write_leda(G, 'graph.leda')
```

## 使用 Pandas

### 從 Pandas DataFrame
```python
import pandas as pd

# 從邊列表 DataFrame 建立圖
df = pd.DataFrame({
    'source': [1, 2, 3, 4],
    'target': [2, 3, 4, 1],
    'weight': [0.5, 1.0, 0.75, 0.25]
})

# 建立圖
G = nx.from_pandas_edgelist(df,
                            source='source',
                            target='target',
                            edge_attr='weight')

# 帶多個邊屬性
G = nx.from_pandas_edgelist(df,
                            source='source',
                            target='target',
                            edge_attr=['weight', 'color', 'type'])

# 建立有向圖
G = nx.from_pandas_edgelist(df,
                            source='source',
                            target='target',
                            create_using=nx.DiGraph())
```

### 轉為 Pandas DataFrame
```python
# 將圖轉換為邊列表 DataFrame
df = nx.to_pandas_edgelist(G)

# 指定邊屬性
df = nx.to_pandas_edgelist(G, source='node1', target='node2')
```

### 使用 Pandas 的鄰接矩陣
```python
# 從鄰接矩陣建立 DataFrame
df = nx.to_pandas_adjacency(G, dtype=int)

# 從鄰接 DataFrame 建立圖
G = nx.from_pandas_adjacency(df)

# 對於有向圖
G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
```

## NumPy 和 SciPy 整合

### 鄰接矩陣
```python
import numpy as np

# 轉為 NumPy 鄰接矩陣
A = nx.to_numpy_array(G, dtype=int)

# 指定節點順序
nodelist = [1, 2, 3, 4, 5]
A = nx.to_numpy_array(G, nodelist=nodelist)

# 從 NumPy 陣列
G = nx.from_numpy_array(A)

# 對於有向圖
G = nx.from_numpy_array(A, create_using=nx.DiGraph())
```

### 稀疏矩陣（SciPy）
```python
from scipy import sparse

# 轉為稀疏矩陣
A = nx.to_scipy_sparse_array(G)

# 指定格式（csr、csc、coo 等）
A_csr = nx.to_scipy_sparse_array(G, format='csr')

# 從稀疏矩陣
G = nx.from_scipy_sparse_array(A)
```

## JSON 格式

### 節點連結格式
```python
import json

# 轉為節點連結格式（適合 d3.js）
data = nx.node_link_data(G)
with open('graph.json', 'w') as f:
    json.dump(data, f)

# 從節點連結格式
with open('graph.json', 'r') as f:
    data = json.load(f)
G = nx.node_link_graph(data)
```

### 鄰接資料格式
```python
# 轉為鄰接格式
data = nx.adjacency_data(G)
with open('graph.json', 'w') as f:
    json.dump(data, f)

# 從鄰接格式
with open('graph.json', 'r') as f:
    data = json.load(f)
G = nx.adjacency_graph(data)
```

### 樹資料格式
```python
# 對於樹形圖
data = nx.tree_data(G, root=0)
with open('tree.json', 'w') as f:
    json.dump(data, f)

# 從樹格式
with open('tree.json', 'r') as f:
    data = json.load(f)
G = nx.tree_graph(data)
```

## Pickle 格式

### 二進位 Pickle
```python
import pickle

# 寫入 pickle（保留所有 Python 物件）
with open('graph.pkl', 'wb') as f:
    pickle.dump(G, f)

# 讀取 pickle
with open('graph.pkl', 'rb') as f:
    G = pickle.load(f)

# NetworkX 便利函數
nx.write_gpickle(G, 'graph.gpickle')
G = nx.read_gpickle('graph.gpickle')
```

## CSV 檔案

### 自訂 CSV 讀取
```python
import csv

# 從 CSV 讀取邊
G = nx.Graph()
with open('edges.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        G.add_edge(row['source'], row['target'], weight=float(row['weight']))

# 將邊寫入 CSV
with open('edges.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['source', 'target', 'weight'])
    for u, v, data in G.edges(data=True):
        writer.writerow([u, v, data.get('weight', 1.0)])
```

## 資料庫整合

### SQL 資料庫
```python
import sqlite3
import pandas as pd

# 透過 pandas 從 SQL 資料庫讀取
conn = sqlite3.connect('network.db')
df = pd.read_sql_query("SELECT source, target, weight FROM edges", conn)
G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='weight')
conn.close()

# 寫入 SQL 資料庫
df = nx.to_pandas_edgelist(G)
conn = sqlite3.connect('network.db')
df.to_sql('edges', conn, if_exists='replace', index=False)
conn.close()
```

## 視覺化用圖格式

### DOT 格式（Graphviz）
```python
# 寫入 DOT 檔案供 Graphviz 使用
nx.drawing.nx_pydot.write_dot(G, 'graph.dot')

# 讀取 DOT 檔案
G = nx.drawing.nx_pydot.read_dot('graph.dot')

# 直接生成圖像（需要 Graphviz）
from networkx.drawing.nx_pydot import to_pydot
pydot_graph = to_pydot(G)
pydot_graph.write_png('graph.png')
```

## Cytoscape 整合

### Cytoscape JSON
```python
# 匯出供 Cytoscape 使用
data = nx.cytoscape_data(G)
with open('cytoscape.json', 'w') as f:
    json.dump(data, f)

# 從 Cytoscape 匯入
with open('cytoscape.json', 'r') as f:
    data = json.load(f)
G = nx.cytoscape_graph(data)
```

## 專門格式

### Matrix Market 格式
```python
from scipy.io import mmread, mmwrite

# 讀取 Matrix Market
A = mmread('graph.mtx')
G = nx.from_scipy_sparse_array(A)

# 寫入 Matrix Market
A = nx.to_scipy_sparse_array(G)
mmwrite('graph.mtx', A)
```

### Shapefile（地理網路）
```python
# 需要 pyshp 函式庫
# 從 shapefile 讀取地理網路
G = nx.read_shp('roads.shp')

# 寫入 shapefile
nx.write_shp(G, 'network')
```

## 格式選擇指南

### 根據需求選擇

**鄰接列表** - 簡單、人類可讀、無屬性
- 最適合：簡單無權圖、快速檢視

**邊列表** - 簡單、支援權重、人類可讀
- 最適合：加權圖、資料匯入/匯出

**GML/GraphML** - 完整屬性保留、基於 XML
- 最適合：帶所有元資料的完整圖序列化

**JSON** - 網頁友好、JavaScript 整合
- 最適合：網頁應用、d3.js 視覺化

**Pickle** - 快速、保留 Python 物件、二進位
- 最適合：僅 Python 儲存、複雜屬性

**Pandas** - 資料分析整合、DataFrame 操作
- 最適合：資料處理管線、統計分析

**NumPy/SciPy** - 數值計算、稀疏矩陣
- 最適合：矩陣運算、科學計算

**DOT** - 視覺化、Graphviz 整合
- 最適合：建立視覺圖表

## 效能考量

### 大型圖
對於大型圖，考慮：
```python
# 使用壓縮格式
import gzip
with gzip.open('graph.adjlist.gz', 'wt') as f:
    nx.write_adjlist(G, f)

with gzip.open('graph.adjlist.gz', 'rt') as f:
    G = nx.read_adjlist(f)

# 使用二進位格式（更快）
nx.write_gpickle(G, 'graph.gpickle')  # 比文字格式更快

# 對鄰接使用稀疏矩陣
A = nx.to_scipy_sparse_array(G, format='csr')  # 記憶體高效
```

### 增量載入
對於非常大的圖：
```python
# 從邊列表增量載入圖
G = nx.Graph()
with open('huge_graph.edgelist') as f:
    for line in f:
        u, v = line.strip().split()
        G.add_edge(u, v)

        # 分塊處理
        if G.number_of_edges() % 100000 == 0:
            print(f"Loaded {G.number_of_edges()} edges")
```

## 錯誤處理

### 穩健的檔案讀取
```python
try:
    G = nx.read_graphml('graph.graphml')
except nx.NetworkXError as e:
    print(f"Error reading GraphML: {e}")
except FileNotFoundError:
    print("File not found")
    G = nx.Graph()

# 檢查是否支援檔案格式
if os.path.exists('graph.txt'):
    with open('graph.txt') as f:
        first_line = f.readline()
        # 檢測格式並相應讀取
```
