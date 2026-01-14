# NetworkX 圖生成器

## 經典圖

### 完全圖
```python
# 完全圖（所有節點相互連接）
G = nx.complete_graph(n=10)

# 完全二部圖
G = nx.complete_bipartite_graph(n1=5, n2=7)

# 完全多部圖
G = nx.complete_multipartite_graph(3, 4, 5)  # 三個分區
```

### 環圖和路徑圖
```python
# 環圖（節點排列成圓形）
G = nx.cycle_graph(n=20)

# 路徑圖（線性鏈）
G = nx.path_graph(n=15)

# 圓形梯子圖
G = nx.circular_ladder_graph(n=10)
```

### 正則圖
```python
# 空圖（無邊）
G = nx.empty_graph(n=10)

# 零圖（無節點）
G = nx.null_graph()

# 星形圖（一個中心節點連接所有其他節點）
G = nx.star_graph(n=19)  # 建立 20 節點星形

# 輪形圖（帶中心軸的環）
G = nx.wheel_graph(n=10)
```

### 特殊命名圖
```python
# 牛頭圖
G = nx.bull_graph()

# Chvatal 圖
G = nx.chvatal_graph()

# 立方體圖
G = nx.cubical_graph()

# 菱形圖
G = nx.diamond_graph()

# 正十二面體圖
G = nx.dodecahedral_graph()

# Heawood 圖
G = nx.heawood_graph()

# 房屋圖
G = nx.house_graph()

# Petersen 圖
G = nx.petersen_graph()

# 空手道俱樂部圖（經典社交網路）
G = nx.karate_club_graph()
```

## 隨機圖

### Erdős-Rényi 圖
```python
# G(n, p) 模型：n 個節點，邊機率 p
G = nx.erdos_renyi_graph(n=100, p=0.1, seed=42)

# G(n, m) 模型：n 個節點，剛好 m 條邊
G = nx.gnm_random_graph(n=100, m=500, seed=42)

# 快速版本（用於大型稀疏圖）
G = nx.fast_gnp_random_graph(n=10000, p=0.0001, seed=42)
```

### Watts-Strogatz 小世界
```python
# 帶重連的小世界網路
# n 個節點，k 個最近鄰，重連機率 p
G = nx.watts_strogatz_graph(n=100, k=6, p=0.1, seed=42)

# 連通版本（保證連通性）
G = nx.connected_watts_strogatz_graph(n=100, k=6, p=0.1, tries=100, seed=42)
```

### Barabási-Albert 優先附著
```python
# 無標度網路（冪律度分佈）
# n 個節點，新節點附著 m 條邊
G = nx.barabasi_albert_graph(n=100, m=3, seed=42)

# 帶參數的擴展版本
G = nx.extended_barabasi_albert_graph(n=100, m=3, p=0.5, q=0.2, seed=42)
```

### 冪律度序列
```python
# 冪律聚類圖
G = nx.powerlaw_cluster_graph(n=100, m=3, p=0.1, seed=42)

# 隨機冪律樹
G = nx.random_powerlaw_tree(n=100, gamma=3, seed=42, tries=1000)
```

### 配置模型
```python
# 指定度序列的圖
degree_sequence = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
G = nx.configuration_model(degree_sequence, seed=42)

# 移除自環和平行邊
G = nx.Graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
```

### 隨機幾何圖
```python
# 單位正方形中的節點，距離 < 半徑則連邊
G = nx.random_geometric_graph(n=100, radius=0.2, seed=42)

# 帶位置
pos = nx.get_node_attributes(G, 'pos')
```

### 隨機正則圖
```python
# 每個節點剛好有 d 個鄰居
G = nx.random_regular_graph(d=3, n=100, seed=42)
```

### 隨機區塊模型
```python
# 社群結構模型
sizes = [50, 50, 50]  # 三個社群
probs = [[0.25, 0.05, 0.02],  # 社群內和社群間機率
         [0.05, 0.35, 0.07],
         [0.02, 0.07, 0.40]]
G = nx.stochastic_block_model(sizes, probs, seed=42)
```

## 格子和網格圖

### 網格圖
```python
# 2D 網格
G = nx.grid_2d_graph(m=5, n=7)  # 5x7 網格

# 3D 網格
G = nx.grid_graph(dim=[5, 7, 3])  # 5x7x3 網格

# 六邊形格子
G = nx.hexagonal_lattice_graph(m=5, n=7)

# 三角形格子
G = nx.triangular_lattice_graph(m=5, n=7)
```

### 超立方體
```python
# n 維超立方體
G = nx.hypercube_graph(n=4)
```

## 樹形圖

### 隨機樹
```python
# n 個節點的隨機樹
G = nx.random_tree(n=100, seed=42)

# 前綴樹（字典樹）
G = nx.prefix_tree([[0, 1, 2], [0, 1, 3], [0, 4]])
```

### 平衡樹
```python
# 高度為 h 的平衡 r 分樹
G = nx.balanced_tree(r=2, h=5)  # 二元樹，高度 5

# n 個節點的完全 r 分樹
G = nx.full_rary_tree(r=3, n=100)  # 三元樹
```

### 啞鈴圖和棒棒糖圖
```python
# 兩個完全圖由路徑連接
G = nx.barbell_graph(m1=5, m2=3)  # 兩個 K_5 圖由 3 節點路徑連接

# 完全圖連接路徑
G = nx.lollipop_graph(m=7, n=5)  # K_7 連接 5 節點路徑
```

## 社交網路模型

### 空手道俱樂部
```python
# Zachary 空手道俱樂部（經典社交網路）
G = nx.karate_club_graph()
```

### 戴維斯南方婦女
```python
# 二部社交網路
G = nx.davis_southern_women_graph()
```

### 佛羅倫斯家族
```python
# 歷史婚姻和商業網路
G = nx.florentine_families_graph()
```

### 悲慘世界
```python
# 角色共現網路
G = nx.les_miserables_graph()
```

## 有向圖生成器

### 隨機有向圖
```python
# 有向 Erdős-Rényi
G = nx.gnp_random_graph(n=100, p=0.1, directed=True, seed=42)

# 無標度有向圖
G = nx.scale_free_graph(n=100, seed=42)
```

### DAG（有向無環圖）
```python
# 隨機 DAG
G = nx.gnp_random_graph(n=20, p=0.2, directed=True, seed=42)
G = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])  # 移除反向邊
```

### 競賽圖
```python
# 隨機競賽（完全有向圖）
G = nx.random_tournament(n=10, seed=42)
```

## 複製分歧模型

### 複製分歧圖
```python
# 生物網路模型（蛋白質交互作用網路）
G = nx.duplication_divergence_graph(n=100, p=0.5, seed=42)
```

## 度序列生成器

### 有效度序列
```python
# 檢查度序列是否有效（可圖化）
sequence = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
is_valid = nx.is_graphical(sequence)

# 對於有向圖
in_sequence = [2, 2, 2, 1, 1]
out_sequence = [2, 2, 1, 2, 1]
is_valid = nx.is_digraphical(in_sequence, out_sequence)
```

### 從度序列建立
```python
# Havel-Hakimi 演算法
G = nx.havel_hakimi_graph(degree_sequence)

# 配置模型（允許多邊/自環）
G = nx.configuration_model(degree_sequence)

# 有向配置模型
G = nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
```

## 二部圖

### 隨機二部圖
```python
# 具有兩個節點集的隨機二部圖
G = nx.bipartite.random_graph(n=50, m=30, p=0.1, seed=42)

# 二部圖的配置模型
G = nx.bipartite.configuration_model(deg1=[3, 3, 2], deg2=[2, 2, 2, 2], seed=42)
```

### 二部圖生成器
```python
# 完全二部圖
G = nx.complete_bipartite_graph(n1=5, n2=7)

# Gnmk 隨機二部圖（n、m 節點，k 條邊）
G = nx.bipartite.gnmk_random_graph(n=10, m=8, k=20, seed=42)
```

## 圖運算子

### 圖運算
```python
# 聯集
G = nx.union(G1, G2)

# 不相交聯集
G = nx.disjoint_union(G1, G2)

# 組合（疊加）
G = nx.compose(G1, G2)

# 補圖
G = nx.complement(G1)

# 笛卡爾積
G = nx.cartesian_product(G1, G2)

# 張量（Kronecker）積
G = nx.tensor_product(G1, G2)

# 強積
G = nx.strong_product(G1, G2)
```

## 自訂和種子

### 設定隨機種子
始終設定種子以獲得可重現的圖：
```python
G = nx.erdos_renyi_graph(n=100, p=0.1, seed=42)
```

### 轉換圖類型
```python
# 轉換為特定類型
G_directed = G.to_directed()
G_undirected = G.to_undirected()
G_multi = nx.MultiGraph(G)
```

## 效能考量

### 快速生成器
對於大型圖，使用優化的生成器：
```python
# 快速 ER 圖（稀疏）
G = nx.fast_gnp_random_graph(n=10000, p=0.0001, seed=42)
```

### 記憶體效率
某些生成器增量建立圖以節省記憶體。對於非常大的圖，考慮：
- 使用稀疏表示
- 根據需要生成子圖
- 使用鄰接列表或邊列表而非完整圖

## 驗證和屬性

### 檢查生成的圖
```python
# 驗證屬性
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G)}")
print(f"Connected: {nx.is_connected(G)}")

# 度分佈
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
```
