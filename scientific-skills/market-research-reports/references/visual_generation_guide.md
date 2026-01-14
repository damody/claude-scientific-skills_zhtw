# 市場研究報告視覺生成指南

市場研究報告中視覺生成的完整提示和指引。

---

## 概述

市場研究報告應從 **5-6 個必要視覺元素** 開始，以建立分析框架。撰寫特定章節時可視需要生成額外視覺元素。本指南提供 `scientific-schematics` 和 `generate-image` 技能的即用提示。

### 核心視覺元素（優先生成 - 優先順序 1-6）

每份市場報告開始時先生成這 5-6 個核心視覺元素：

1. **市場成長軌跡圖** - 顯示市場規模趨勢
2. **TAM/SAM/SOM 圖** - 市場機會分解
3. **波特五力分析** - 競爭動態框架
4. **競爭定位矩陣** - 策略定位
5. **風險熱力圖** - 風險評估視覺化
6. **執行摘要資訊圖表**（選擇性）- 報告概覽

### 延伸視覺元素（視需要生成 - 優先順序 7+）

撰寫過程中當特定章節需要視覺支援時可生成額外視覺元素：
- 區域分布圖
- 區隔分析
- 客戶旅程圖
- 技術路線圖
- 法規時間軸
- 財務預測
- 實施時程表

### 工具選擇

| 視覺類型 | 工具 | 理由 |
|-------------|------|-----------|
| 圖表（長條、折線、圓餅） | scientific-schematics | 精確資料呈現 |
| 圖解（流程、結構） | scientific-schematics | 清晰技術版面 |
| 矩陣（2x2、定位） | scientific-schematics | 策略框架 |
| 時間軸 | scientific-schematics | 順序資訊 |
| 資訊圖表 | generate-image | 創意視覺綜合 |
| 概念插圖 | generate-image | 抽象概念 |

---

## 視覺命名慣例

### 核心視覺元素（優先生成）
```
figures/
├── 01_market_growth_trajectory.png      # 優先順序 1
├── 02_tam_sam_som.png                   # 優先順序 2
├── 03_porters_five_forces.png           # 優先順序 3
├── 04_competitive_positioning.png       # 優先順序 4
├── 05_risk_heatmap.png                  # 優先順序 5
└── 06_exec_summary_infographic.png      # 優先順序 6（選擇性）
```

### 延伸視覺元素（視需要生成）
```
figures/
├── 07_industry_ecosystem.png
├── 08_regional_breakdown.png
├── 09_segment_growth.png
├── 10_driver_impact_matrix.png
├── 11_pestle_analysis.png
├── 12_trends_timeline.png
├── 13_market_share.png
├── 14_strategic_groups.png
├── 15_customer_segments.png
├── 16_segment_attractiveness.png
├── 17_customer_journey.png
├── 18_technology_roadmap.png
├── 19_innovation_curve.png
├── 20_regulatory_timeline.png
├── 21_risk_mitigation.png
├── 22_opportunity_matrix.png
├── 23_recommendation_priority.png
├── 24_implementation_timeline.png
├── 25_milestone_tracker.png
├── 26_financial_projections.png
└── 27_scenario_analysis.png
```

---

## 核心視覺元素（優先順序 1-6）- 優先生成這些

### 優先順序 1：市場成長軌跡圖

**工具：** scientific-schematics

**目的：** 顯示歷史和預測市場規模的基礎視覺元素

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Bar chart market growth 2020 to 2034. Historical bars 2020-2024 in dark blue, projected bars 2025-2034 in light blue. Y-axis billions USD, X-axis years. CAGR annotation. Data labels on each bar. Vertical dashed line between 2024 and 2025. Title: Market Growth Trajectory. Professional white background" \
  -o figures/01_market_growth_trajectory.png --doc-type report
```

---

### 優先順序 2：TAM/SAM/SOM 圖

**工具：** scientific-schematics

**目的：** 市場機會規模視覺化

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "TAM SAM SOM concentric circles. Outer circle TAM Total Addressable Market. Middle circle SAM Serviceable Addressable Market. Inner circle SOM Serviceable Obtainable Market. Each labeled with acronym, full name, placeholder for dollar value. Arrows pointing to each with descriptions. Blue gradient darkest outer to lightest inner. White background professional appearance" \
  -o figures/02_tam_sam_som.png --doc-type report
```

---

### 優先順序 3：波特五力分析圖

**工具：** scientific-schematics

**目的：** 競爭動態框架

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Porter's Five Forces diagram. Center box Competitive Rivalry with rating. Four surrounding boxes with arrows to center: Top Threat of New Entrants, Left Bargaining Power Suppliers, Right Bargaining Power Buyers, Bottom Threat of Substitutes. Color code HIGH red, MEDIUM yellow, LOW green. Include 2-3 key factors per box. Professional appearance" \
  -o figures/03_porters_five_forces.png --doc-type report
```

---

### 優先順序 4：競爭定位矩陣

**工具：** scientific-schematics

**目的：** 主要市場參與者的策略定位

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 competitive positioning matrix. X-axis Market Focus Niche to Broad. Y-axis Solution Approach Product to Platform. Quadrants: Upper-right Platform Leaders, Upper-left Niche Platforms, Lower-right Product Leaders, Lower-left Specialists. Plot 8-10 company circles with names. Circle size = market share. Legend for sizes. Professional appearance" \
  -o figures/04_competitive_positioning.png --doc-type report
```

---

### 優先順序 5：風險熱力圖

**工具：** scientific-schematics

**目的：** 視覺化風險評估矩陣

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Risk heatmap matrix. X-axis Impact Low Medium High Critical. Y-axis Probability Unlikely Possible Likely Very Likely. Cell colors: Green low risk, Yellow medium, Orange high, Red critical. Plot 10-12 numbered risks R1 R2 etc as labeled points. Legend with risk names. Professional clear" \
  -o figures/05_risk_heatmap.png --doc-type report
```

---

### 優先順序 6：執行摘要資訊圖表（選擇性）

**工具：** generate-image

**目的：** 封面或執行摘要的高層次視覺綜合

**命令：**
```bash
python skills/generate-image/scripts/generate_image.py \
  "Executive summary infographic for market research, one page layout, central large metric showing market size, four quadrants showing growth rate key players top segments regional leaders, modern flat design, professional blue and green color scheme, clean white background, corporate business aesthetic" \
  --output figures/06_exec_summary_infographic.png
```

---

## 延伸視覺元素 - 撰寫過程中視需要生成

以下視覺元素可在撰寫需要這些元素的特定章節時生成。

---

## 前置部分視覺元素

### 延伸：封面圖像 / 主視覺

**工具：** generate-image

**提示：**
```
[市場名稱] 市場研究報告的專業執行摘要資訊圖表。
展示關鍵指標的現代資料視覺化風格：市場規模、成長率、主要參與者。
符合企業設計的藍綠配色方案。
帶有圖示的簡潔極簡設計。
高解析度、出版品質。
無文字覆蓋，僅圖像。
```

**命令：**
```bash
python skills/generate-image/scripts/generate_image.py \
  "Professional executive summary infographic for [MARKET] market research report, modern data visualization style, key metrics display, blue and green corporate color scheme, clean minimalist design with icons, high resolution publication quality" \
  --output figures/01_cover_image.png
```

### 2. 執行摘要資訊圖表

**工具：** generate-image

**提示：**
```
單頁執行摘要資訊圖表，顯示：
- 大型中央指標：$XX 億市場規模
- 四個象限：成長率、主要參與者、頂級區隔、區域領導者
- 帶有資料視覺化元素的現代扁平設計
- 專業藍色（#003366）和綠色（#008060）配色方案
- 乾淨的白色背景
- 商業/企業美學
```

**命令：**
```bash
python skills/generate-image/scripts/generate_image.py \
  "Executive summary infographic for market research, one page layout, central large metric showing market size, four quadrants showing growth rate key players top segments regional leaders, modern flat design, professional blue and green color scheme, clean white background, corporate business aesthetic" \
  --output figures/02_exec_summary_infographic.png
```

---

## 第一章：市場概述視覺元素

### 3. 產業生態系統圖

**工具：** scientific-schematics

**提示：**
```
產業生態系統價值鏈圖，顯示從左到右的水平流動：
[供應商/投入] → [製造商/加工商] → [經銷商/通路] → [終端使用者/客戶]

各階段下方顯示 3-4 個範例參與者類型的小方塊。
用箭頭顯示產品/服務流（實線）和資金流（虛線）。
在鏈條上方包含監管機構作為監督層。
專業藍色配色方案。
乾淨的白色背景。
所有文字清晰可讀。
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Industry ecosystem value chain diagram. Horizontal flow left to right: Suppliers box → Manufacturers box → Distributors box → End Users box. Below each main box show 3-4 smaller boxes with example player types. Solid arrows for product flow, dashed arrows for money flow. Regulatory oversight layer above. Professional blue color scheme, white background, clear labels" \
  -o figures/03_industry_ecosystem.png --doc-type report
```

### 4. 市場結構圖

**工具：** scientific-schematics

**提示：**
```
市場結構圖，顯示同心矩形：
- 中心：核心市場（標註市場名稱）
- 第二層：相鄰市場（標註 4-5 個相鄰市場名稱）
- 第三層：賦能技術（標註關鍵技術）
- 外層：法規框架

各層使用不同深淺的藍色。
包含關鍵元素的小圖示或標籤。
專業外觀。
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Market structure diagram with concentric rectangles. Center: Core Market [MARKET NAME]. Second layer: Adjacent Markets with 4-5 labels. Third layer: Enabling Technologies with key tech labels. Outer layer: Regulatory Framework. Different blue shades for each layer, professional appearance, clear labels" \
  -o figures/03b_market_structure.png --doc-type report
```

---

## 第二章：市場規模與成長視覺元素

### 5. 市場成長軌跡圖

**工具：** scientific-schematics

**提示：**
```
長條圖顯示 2020 至 2034 年市場成長。
歷史年份（2020-2024）：深藍色長條
預測年份（2025-2034）：淺藍色長條
Y 軸：市場規模（十億美元，0 至 $XXX）
X 軸：年份
包含年複合成長率標註「XX.X% CAGR (2024-2034)」
各長條頂部有資料標籤
垂直虛線分隔歷史和預測
標題：「[市場名稱] 市場成長軌跡」
專業外觀，白色背景
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Bar chart market growth 2020 to 2034. Historical bars 2020-2024 in dark blue, projected bars 2025-2034 in light blue. Y-axis billions USD, X-axis years. CAGR annotation XX.X% (2024-2034). Data labels on each bar. Vertical dashed line between 2024 and 2025. Title: Market Growth Trajectory. Professional white background" \
  -o figures/04_market_growth_trajectory.png --doc-type report
```

### 6. TAM/SAM/SOM 圖

**工具：** scientific-schematics

**提示：**
```
TAM SAM SOM 同心圓圖：
- 外圈：TAM（總潛在市場）- $XXX 億
- 中圈：SAM（可服務潛在市場）- $XX 億
- 內圈：SOM（可獲得服務市場）- $X 億

各圈標註：
- 粗體縮寫
- 完整名稱
- 金額

箭頭指向各圈並附描述
使用藍色漸層（TAM 最深，SOM 最淺）
專業外觀
白色背景
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "TAM SAM SOM concentric circles. Outer circle TAM Total Addressable Market [VALUE]B. Middle circle SAM Serviceable Addressable Market [VALUE]B. Inner circle SOM Serviceable Obtainable Market [VALUE]B. Each labeled with acronym, full name, dollar value. Arrows pointing to each with descriptions. Blue gradient darkest outer to lightest inner. White background professional" \
  -o figures/05_tam_sam_som.png --doc-type report
```

### 7. 區域市場分布

**工具：** scientific-schematics

**提示：**
```
圓餅圖或樹狀圖顯示區域市場分布：
- 北美：XX%（$X.XB）- 深藍色
- 歐洲：XX%（$X.XB）- 中藍色
- 亞太地區：XX%（$X.XB）- 藍綠色
- 拉丁美洲：X%（$X.XB）- 淺藍色
- 中東和非洲：X%（$X.XB）- 灰藍色

各區域包含百分比和金額
右側圖例
標題：「各區域市場規模（2024 年）」
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Pie chart regional market breakdown. North America XX% dark blue, Europe XX% medium blue, Asia-Pacific XX% teal, Latin America XX% light blue, Middle East Africa XX% gray blue. Show percentage and dollar value for each slice. Legend on right. Title: Market Size by Region 2024. Professional appearance" \
  -o figures/06_regional_breakdown.png --doc-type report
```

### 8. 區隔成長比較

**工具：** scientific-schematics

**提示：**
```
水平長條圖比較區隔成長率：
- Y 軸：區隔名稱（5-7 個區隔）
- X 軸：年複合成長率百分比（0% 至 30%）
- 長條按成長率著色：綠色（最高）至藍色（最低）
- 各長條顯示確切百分比資料標籤
- 按成長率從高到低排序
- 標題：「區隔成長率比較（2024-2034 年複合成長率）」
- 包含平均線或標記
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Horizontal bar chart segment growth comparison. Y-axis 5-7 segment names, X-axis CAGR percentage 0-30%. Bars colored green highest to blue lowest. Data labels with exact percentages. Sorted highest to lowest. Title: Segment Growth Rate Comparison CAGR 2024-2034. Include market average line" \
  -o figures/07_segment_growth.png --doc-type report
```

---

## 第三章：產業驅動因素與趨勢視覺元素

### 9. 驅動因素影響矩陣

**工具：** scientific-schematics

**提示：**
```
市場驅動因素評估的 2x2 矩陣：
- X 軸：對市場的影響（低 → 高）
- Y 軸：發生機率（低 → 高）
- 右上象限：「關鍵驅動因素」（紅色/橙色背景）
- 左上象限：「監測」（黃色背景）
- 右下象限：「仔細觀察」（黃色背景）
- 左下象限：「較低優先順序」（綠色背景）

繪製 8-10 個驅動因素為標註圓圈：
- 圓圈大小代表目前市場影響
- 位置基於評級

包含圓圈大小圖例
專業外觀並附清晰標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 matrix driver impact assessment. X-axis Impact Low to High, Y-axis Probability Low to High. Quadrants: Upper-right CRITICAL DRIVERS red, Upper-left MONITOR yellow, Lower-right WATCH CAREFULLY yellow, Lower-left LOWER PRIORITY green. Plot 8-10 labeled driver circles at appropriate positions. Circle size indicates current impact. Professional clear labels" \
  -o figures/08_driver_impact_matrix.png --doc-type report
```

### 10. PESTLE 分析圖

**工具：** scientific-schematics

**提示：**
```
PESTLE 分析六邊形圖：
- 中心六邊形：「[市場名稱]」
- 連接到中心的六個周圍六邊形：
  - 政治（紅色/橙色）
  - 經濟（藍色）
  - 社會（綠色）
  - 技術（橙色）
  - 法律（紫色）
  - 環境（藍綠色）

各外圍六邊形包含 2-3 個關鍵要點
中心與外圍六邊形之間有連接線
專業外觀
各六邊形內文字清晰可讀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "PESTLE hexagonal diagram. Center hexagon labeled MARKET. Six surrounding hexagons: Political red, Economic blue, Social green, Technological orange, Legal purple, Environmental teal. Each outer hexagon has 2-3 bullet points of key factors. Lines connecting center to each. Professional appearance clear readable text" \
  -o figures/09_pestle_analysis.png --doc-type report
```

### 11. 產業趨勢時間軸

**工具：** scientific-schematics

**提示：**
```
水平時間軸顯示 2024 至 2030 年新興趨勢：
- 主要水平軸帶有年份標記
- 在時間軸不同位置繪製 6-8 個趨勢
- 各趨勢顯示：
  - 圖示或符號
  - 趨勢名稱
  - 下方 3-5 字簡短描述

按趨勢類別色彩編碼：
- 技術趨勢：藍色
- 市場趨勢：綠色
- 法規趨勢：橙色

在 2024 年包含「現在」標記
專業外觀並附清晰標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Horizontal timeline 2024 to 2030. Plot 6-8 emerging trends at different years. Each trend with icon, name, brief description. Color code: Technology trends blue, Market trends green, Regulatory trends orange. Current marker at 2024. Professional clear labels" \
  -o figures/10_trends_timeline.png --doc-type report
```

---

## 第四章：競爭格局視覺元素

### 12. 波特五力分析圖

**工具：** scientific-schematics

**提示：**
```
波特五力分析圖，中心和四個周圍方塊：

中心方塊：「競爭對抗」及評級 [高/中/低]

周圍方塊以箭頭連接：
- 上方：「新進入者威脅」[評級]
- 左方：「供應商議價能力」[評級]
- 右方：「買家議價能力」[評級]
- 下方：「替代品威脅」[評級]

評級色彩編碼：
- 高：紅色/橙色背景
- 中：黃色背景
- 低：綠色背景

箭頭指向中心
各方塊包含關鍵因素要點
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Porter's Five Forces diagram. Center box Competitive Rivalry [RATING]. Four surrounding boxes with arrows to center: Top Threat of New Entrants [RATING], Left Bargaining Power Suppliers [RATING], Right Bargaining Power Buyers [RATING], Bottom Threat of Substitutes [RATING]. Color code HIGH red, MEDIUM yellow, LOW green. Include 2-3 key factors per box. Professional appearance" \
  -o figures/11_porters_five_forces.png --doc-type report
```

### 13. 市場份額圖

**工具：** scientific-schematics

**提示：**
```
圓餅圖或甜甜圈圖顯示市場份額：
- 前 10 名公司使用不同顏色
- 公司 A：XX%（最大塊，深藍色）
- 公司 B：XX%（中藍色）
- [繼續前 10 名]
- 其他：XX%（灰色）

包含：
- 各塊百分比標籤
- 圖例或塊上的公司名稱
- 總市場規模標註
- 標題：「各公司市場份額（2024 年）」

專業外觀
色盲友善調色盤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Pie chart market share top 10 companies. Company A XX% dark blue, Company B XX% medium blue, [list companies and shares], Others XX% gray. Percentage labels on slices. Legend with company names. Total market size annotation. Title: Market Share by Company 2024. Colorblind-friendly colors professional" \
  -o figures/12_market_share.png --doc-type report
```

### 14. 競爭定位矩陣

**工具：** scientific-schematics

**提示：**
```
2x2 競爭定位矩陣：
- X 軸：市場聚焦（利基 ← → 廣泛）
- Y 軸：解決方案類型（產品 ← → 平台）

象限標籤：
- 右上：「平台領導者」
- 左上：「利基平台」
- 右下：「產品領導者」
- 左下：「專家」

繪製 8-10 家公司為標註圓圈：
- 圓圈大小代表市場份額
- 位置基於策略

包含圓圈大小圖例
公司名稱標籤
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 competitive positioning matrix. X-axis Market Focus Niche to Broad. Y-axis Solution Approach Product to Platform. Quadrants: Upper-right Platform Leaders, Upper-left Niche Platforms, Lower-right Product Leaders, Lower-left Specialists. Plot 8-10 company circles with names. Circle size = market share. Legend for sizes. Professional" \
  -o figures/13_competitive_positioning.png --doc-type report
```

### 15. 策略群組圖

**工具：** scientific-schematics

**提示：**
```
策略群組圖顯示競爭者群集：
- X 軸：地理範圍（區域 ← → 全球）
- Y 軸：產品廣度（窄 ← → 寬）

繪製 4-5 個橢圓「氣泡」代表策略群組：
- 各氣泡包含 2-4 個公司名稱
- 氣泡大小代表群組的集體市場份額
- 各策略群組使用不同顏色

標註各策略群組：
- 「全球通才」
- 「區域專家」
- 「專注創新者」
- 等等

專業外觀
公司名稱標籤清晰
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Strategic group map. X-axis Geographic Scope Regional to Global. Y-axis Product Breadth Narrow to Broad. Draw 4-5 oval bubbles for strategic groups. Each bubble contains 2-4 company names. Bubble size = collective market share. Label groups: Global Generalists, Regional Specialists, Focused Innovators etc. Different colors per group. Professional clear labels" \
  -o figures/14_strategic_groups.png --doc-type report
```

---

## 第五章：客戶分析視覺元素

### 16. 客戶區隔分布

**工具：** scientific-schematics

**提示：**
```
樹狀圖或圓餅圖顯示客戶區隔：
- 大型企業：XX%（深藍色）
- 中型市場：XX%（中藍色）
- 中小企業：XX%（淺藍色）
- 消費者：XX%（藍綠色）

大小代表市場份額
各區隔包含：
- 區隔名稱
- 百分比
- 金額

標題：「按市場份額的客戶區隔」
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Treemap customer segmentation. Large Enterprise XX% dark blue, Mid-Market XX% medium blue, SMB XX% light blue, Consumer XX% teal. Each segment shows name percentage dollar value. Title: Customer Segmentation by Market Share. Professional appearance" \
  -o figures/15_customer_segments.png --doc-type report
```

### 17. 區隔吸引力矩陣

**工具：** scientific-schematics

**提示：**
```
2x2 區隔吸引力矩陣：
- X 軸：區隔規模（小 ← → 大）
- Y 軸：成長率（低 ← → 高）

象限標籤和行動：
- 右上：「優先 - 大力投資」
- 左上：「投資成長」
- 右下：「收割」
- 左下：「降低優先順序」

繪製客戶區隔為標註圓圈
圓圈大小代表獲利能力
各區隔使用不同顏色
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 segment attractiveness matrix. X-axis Segment Size Small to Large. Y-axis Growth Rate Low to High. Quadrants: Upper-right PRIORITY Invest Heavily, Upper-left INVEST TO GROW, Lower-right HARVEST, Lower-left DEPRIORITIZE. Plot customer segments as circles. Circle size = profitability. Different colors. Professional" \
  -o figures/16_segment_attractiveness.png --doc-type report
```

### 18. 客戶旅程圖

**工具：** scientific-schematics

**提示：**
```
客戶旅程水平流程圖顯示 5-6 個階段：
認知 → 考慮 → 決策 → 實施 → 使用 → 擁護

各階段下方顯示三行：
1. 關鍵活動（客戶做什麼）
2. 痛點（面臨的挑戰）
3. 接觸點（如何互動）

各階段使用圖示
顏色漸層隨旅程進展從淺到深
專業外觀
清晰標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Customer journey horizontal flowchart. 5 stages left to right: Awareness, Consideration, Decision, Implementation, Usage, Advocacy. Each stage shows Key Activities, Pain Points, Touchpoints in rows below. Icons for each stage. Color gradient light to dark. Professional clear labels" \
  -o figures/17_customer_journey.png --doc-type report
```

---

## 第六章：技術格局視覺元素

### 19. 技術路線圖

**工具：** scientific-schematics

**提示：**
```
技術路線圖時間軸從 2024 至 2030 年：
三條平行水平軌道：
1. 核心技術（藍色）- 目前基礎
2. 新興技術（綠色）- 發展中能力
3. 賦能技術（橙色）- 基礎設施/支援

各軌道顯示里程碑和技術引入標記
垂直線連接跨軌道相關技術
各年時間軸標記
技術名稱標註於引入點
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Technology roadmap 2024 to 2030. Three parallel horizontal tracks: Core Technology blue, Emerging Technology green, Enabling Technology orange. Milestones and tech introductions marked on each track. Vertical lines connect related tech. Year markers. Technology names labeled. Professional appearance" \
  -o figures/18_technology_roadmap.png --doc-type report
```

### 20. 創新/採用曲線

**工具：** scientific-schematics

**提示：**
```
Gartner 技術成熟度曲線或技術採用曲線：
從左到右五個階段：
1. 創新觸發（上升）
2. 過度期望峰值（頂峰）
3. 幻滅低谷（底部）
4. 啟蒙坡道（上升）
5. 生產力高原（穩定）

在曲線不同位置繪製 6-8 項技術
各技術標註名稱
按技術類別色彩編碼
專業外觀
清晰軸線標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Gartner Hype Cycle curve. Five phases: Innovation Trigger rising, Peak of Inflated Expectations at top, Trough of Disillusionment at bottom, Slope of Enlightenment rising, Plateau of Productivity stable. Plot 6-8 technologies on curve with labels. Color by category. Professional clear labels" \
  -o figures/19_innovation_curve.png --doc-type report
```

---

## 第七章：法規環境視覺元素

### 21. 法規時間軸

**工具：** scientific-schematics

**提示：**
```
法規時間軸從 2020 至 2028 年：
帶有年份標記的水平時間軸
標記關鍵法規事件：
- 過去法規（深藍色標記，實線）
- 目前法規（當年綠色標記）
- 即將實施法規（淺藍色標記，虛線）

各標記顯示：
- 法規名稱
- 生效日期
- 簡短描述（5-7 字）

當年（2024 年）垂直「現在」線
如涉及多個司法管轄區則按區域分組
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Regulatory timeline 2020 to 2028. Past regulations dark blue solid markers, current green marker, upcoming light blue dashed. Each shows regulation name, date, brief description. Vertical NOW line at 2024. Professional appearance clear labels" \
  -o figures/20_regulatory_timeline.png --doc-type report
```

---

## 第八章：風險分析視覺元素

### 22. 風險熱力圖

**工具：** scientific-schematics

**提示：**
```
風險評估熱力圖/矩陣：
- X 軸：影響（低 → 中 → 高 → 關鍵）
- Y 軸：機率（不太可能 → 可能 → 很可能 → 非常可能）

儲存格顏色漸層：
- 綠色：低風險（低機率，低影響）
- 黃色：中風險
- 橙色：高風險
- 紅色：關鍵風險（高機率，高影響）

在適當儲存格繪製 10-12 個風險為標註點/圓圈
風險標籤清晰可讀
包含風險編號（R1、R2 等）
圖例連結編號和風險名稱
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Risk heatmap matrix. X-axis Impact Low Medium High Critical. Y-axis Probability Unlikely Possible Likely Very Likely. Cell colors: Green low risk, Yellow medium, Orange high, Red critical. Plot 10-12 numbered risks R1 R2 etc as labeled points. Legend with risk names. Professional clear" \
  -o figures/21_risk_heatmap.png --doc-type report
```

### 23. 風險緩解框架

**工具：** scientific-schematics

**提示：**
```
風險緩解圖顯示風險及其緩解措施：
左欄：風險（紅色/橙色方塊）
右欄：緩解策略（綠色/藍色方塊）

用箭頭連接各風險與其緩解措施
按類別分組（市場、法規、技術等）
包含預防和回應策略

風險嚴重程度由方塊顏色深淺表示
專業外觀
清晰標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Risk mitigation diagram. Left column risks in orange/red boxes. Right column mitigation strategies in green/blue boxes. Arrows connecting risks to mitigations. Group by category. Risk severity by color intensity. Include prevention and response. Professional clear labels" \
  -o figures/22_risk_mitigation.png --doc-type report
```

---

## 第九章：策略建議視覺元素

### 24. 機會矩陣

**工具：** scientific-schematics

**提示：**
```
2x2 機會評估矩陣：
- X 軸：市場吸引力（低 ← → 高）
- Y 軸：獲勝能力（低 ← → 高）

象限標籤和策略：
- 右上：「積極追求」（綠色）
- 左上：「建立能力」（黃色）
- 右下：「選擇性投資」（黃色）
- 左下：「避免/退出」（紅色）

繪製 6-8 個機會為標註圓圈
圓圈大小代表機會規模（$）
包含機會名稱
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 opportunity matrix. X-axis Market Attractiveness Low to High. Y-axis Ability to Win Low to High. Quadrants: Upper-right PURSUE AGGRESSIVELY green, Upper-left BUILD CAPABILITIES yellow, Lower-right SELECTIVE INVESTMENT yellow, Lower-left AVOID red. Plot 6-8 opportunity circles with labels. Size = opportunity value. Professional" \
  -o figures/23_opportunity_matrix.png --doc-type report
```

### 25. 建議優先順序矩陣

**工具：** scientific-schematics

**提示：**
```
建議的 2x2 優先順序矩陣：
- X 軸：努力程度/投資（低 ← → 高）
- Y 軸：影響/價值（低 ← → 高）

象限標籤：
- 左上：「速贏」（綠色）- 優先執行
- 右上：「重大專案」（藍色）- 謹慎規劃
- 左下：「填補項目」（灰色）- 有時間再做
- 右下：「吃力不討好」（紅色）- 避免

繪製 6-8 項建議為標註點
建議按優先順序編號
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "2x2 priority matrix. X-axis Effort Low to High. Y-axis Impact Low to High. Quadrants: Upper-left QUICK WINS green Do First, Upper-right MAJOR PROJECTS blue Plan Carefully, Lower-left FILL-INS gray Do If Time, Lower-right THANKLESS TASKS red Avoid. Plot 6-8 numbered recommendations. Professional" \
  -o figures/24_recommendation_priority.png --doc-type report
```

---

## 第十章：實施路線圖視覺元素

### 26. 實施時程表/甘特圖

**工具：** scientific-schematics

**提示：**
```
甘特圖風格實施時程表跨 24 個月：
四個階段顯示為水平長條：
- 第一階段：奠基（第 1-6 月）- 深藍色
- 第二階段：建置（第 4-12 月）- 中藍色
- 第三階段：規模化（第 10-18 月）- 藍綠色
- 第四階段：優化（第 16-24 月）- 淺藍色

階段如日期所示重疊
關鍵里程碑以菱形標記在時間軸上
X 軸月份標記
Y 軸階段名稱
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Gantt chart implementation 24 months. Phase 1 Foundation months 1-6 dark blue. Phase 2 Build months 4-12 medium blue. Phase 3 Scale months 10-18 teal. Phase 4 Optimize months 16-24 light blue. Overlapping bars. Key milestones as diamonds. Month markers X-axis. Professional" \
  -o figures/25_implementation_timeline.png --doc-type report
```

### 27. 里程碑追蹤器

**工具：** scientific-schematics

**提示：**
```
里程碑追蹤器在水平時間軸上顯示 8-10 個關鍵里程碑：
各里程碑顯示：
- 日期/月份
- 里程碑名稱
- 狀態指標：
  - 已完成：綠色勾選 ✓
  - 進行中：黃色圓圈 ○
  - 即將到來：灰色圓圈 ○

按階段分組
時間軸線連接里程碑
時間軸上方包含階段標籤
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Milestone tracker horizontal timeline 8-10 milestones. Each shows date, name, status: Completed green check, In Progress yellow circle, Upcoming gray circle. Group by phase. Phase labels above. Connected timeline line. Professional" \
  -o figures/26_milestone_tracker.png --doc-type report
```

---

## 第十一章：投資論點視覺元素

### 28. 財務預測圖

**工具：** scientific-schematics

**提示：**
```
組合長條圖和折線圖顯示 5 年財務預測：
- 長條圖：各年營收（主 Y 軸，百萬美元）
- 折線圖：成長率疊加（次 Y 軸，百分比）

三種情境顯示：
- 保守：灰色長條
- 基本情境：藍色長條
- 樂觀：綠色長條

X 軸：第 1 年至第 5 年
長條包含資料標籤
情境和成長線圖例
標題：「財務預測（5 年）」
專業外觀
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Combined bar and line chart 5-year projections. Bar chart revenue primary Y-axis dollars. Line chart growth rate secondary Y-axis percent. Three scenarios: Conservative gray, Base Case blue, Optimistic green. X-axis Year 1-5. Data labels. Legend. Title Financial Projections 5-Year. Professional" \
  -o figures/27_financial_projections.png --doc-type report
```

### 29. 情境分析比較

**工具：** scientific-schematics

**提示：**
```
分組長條圖比較三種情境的關鍵指標：
X 軸：指標（第 5 年營收、第 5 年 EBITDA、市場份額、投資報酬率）
Y 軸：數值（各指標適當刻度）

各指標三根長條：
- 保守：灰色
- 基本情境：藍色
- 樂觀：綠色

各長條資料標籤
情境圖例
標題：「情境分析比較」
專業外觀
清晰指標標籤
```

**命令：**
```bash
python skills/scientific-schematics/scripts/generate_schematic.py \
  "Grouped bar chart scenario comparison. X-axis metrics: Revenue Y5, EBITDA Y5, Market Share, ROI. Three bars per metric: Conservative gray, Base Case blue, Optimistic green. Data labels. Legend. Title Scenario Analysis Comparison. Professional clear labels" \
  -o figures/28_scenario_analysis.png --doc-type report
```

---

## 批次生成腳本

為方便起見，使用 `generate_market_visuals.py` 腳本批次生成視覺元素：

```bash
# 僅生成核心 5-6 個視覺元素（建議用於開始報告）
python skills/market-research-reports/scripts/generate_market_visuals.py \
  --topic "Electric Vehicle Charging Infrastructure" \
  --output-dir figures/

# 生成全部 27 個視覺元素（核心 + 延伸，完整涵蓋）
python skills/market-research-reports/scripts/generate_market_visuals.py \
  --topic "Electric Vehicle Charging Infrastructure" \
  --output-dir figures/ \
  --all

# 跳過已生成的檔案
python skills/market-research-reports/scripts/generate_market_visuals.py \
  --topic "Your Market" \
  --output-dir figures/ \
  --skip-existing
```

**預設行為**：僅生成 5-6 個核心優先視覺元素。如需所有章節的全面視覺涵蓋，請使用 `--all` 旗標。

---

## 品質檢核清單

將視覺元素納入報告前確認：

- [ ] 所有文字在預定尺寸下可讀
- [ ] 所有視覺元素顏色一致
- [ ] 配色方案色盲友善
- [ ] 資料標籤準確
- [ ] 圖例清晰完整
- [ ] 標題具描述性
- [ ] 適當處標註來源
- [ ] 解析度 300 DPI 或更高
- [ ] 檔案格式為 PNG
- [ ] 遵循命名慣例
