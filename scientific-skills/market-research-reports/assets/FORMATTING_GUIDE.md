# 市場研究報告格式指南

使用 `market_research.sty` 樣式套件的快速參考。

## 色彩調色盤

### 主要顏色
| 顏色名稱 | RGB | Hex | 用途 |
|------------|-----|-----|-------|
| `primaryblue` | (0, 51, 102) | `#003366` | 標題、題目、連結 |
| `secondaryblue` | (51, 102, 153) | `#336699` | 小節、次要元素 |
| `lightblue` | (173, 216, 230) | `#ADD8E6` | 關鍵洞察方塊背景 |
| `accentblue` | (0, 120, 215) | `#0078D7` | 強調亮點、機會方塊 |

### 次要顏色
| 顏色名稱 | RGB | Hex | 用途 |
|------------|-----|-----|-------|
| `accentgreen` | (0, 128, 96) | `#008060` | 市場資料方塊、正面指標 |
| `lightgreen` | (200, 230, 201) | `#C8E6C9` | 市場資料方塊背景 |
| `warningorange` | (255, 140, 0) | `#FF8C00` | 風險方塊、警告 |
| `alertred` | (198, 40, 40) | `#C62828` | 關鍵風險 |
| `recommendpurple` | (103, 58, 183) | `#673AB7` | 建議方塊 |

### 中性顏色
| 顏色名稱 | RGB | Hex | 用途 |
|------------|-----|-----|-------|
| `darkgray` | (66, 66, 66) | `#424242` | 內文文字 |
| `mediumgray` | (117, 117, 117) | `#757575` | 次要文字 |
| `lightgray` | (240, 240, 240) | `#F0F0F0` | 背景、說明方塊 |
| `tablealt` | (245, 247, 250) | `#F5F7FA` | 表格交替列 |

---

## 方塊環境

### 關鍵洞察方塊（藍色）
用於重大發現、洞察和重要發現。

```latex
\begin{keyinsightbox}[自訂標題]
市場預計到 2030 年將以 15.3% 的年複合成長率成長，
由企業採用增加和有利的法規條件所驅動。
\end{keyinsightbox}
```

### 市場資料方塊（綠色）
用於市場統計、指標和資料亮點。

```latex
\begin{marketdatabox}[市場概況]
\begin{itemize}
    \item \textbf{市場規模（2024 年）：} \marketsize{452 億}
    \item \textbf{預測規模（2030 年）：} \marketsize{987 億}
    \item \textbf{年複合成長率：} \growthrate{15.3}
\end{itemize}
\end{marketdatabox}
```

### 風險方塊（橙色/警告）
用於風險因素、警告和注意事項。

```latex
\begin{riskbox}[市場風險]
歐盟的法規變更可能在未來 18 個月內
影響 40% 的市場參與者。
\end{riskbox}
```

### 關鍵風險方塊（紅色）
用於高嚴重性或關鍵風險。

```latex
\begin{criticalriskbox}[關鍵：供應鏈中斷]
重大供應鏈中斷可能導致 6-12 個月的延遲
以及 30% 的成本增加。
\end{criticalriskbox}
```

### 建議方塊（紫色）
用於策略建議和行動項目。

```latex
\begin{recommendationbox}[策略建議]
\begin{enumerate}
    \item 優先進入亞太地區市場
    \item 與當地經銷商建立策略合作夥伴關係
    \item 投資產品在地化
\end{enumerate}
\end{recommendationbox}
```

### 說明方塊（灰色）
用於定義、註釋和補充資訊。

```latex
\begin{calloutbox}[定義：TAM]
總潛在市場（TAM）代表如果達到 100% 市場份額
時可獲得的總營收機會。
\end{calloutbox}
```

### 執行摘要方塊
執行摘要亮點的特殊樣式。

```latex
\begin{executivesummarybox}[執行摘要]
報告的主要發現和亮點...
\end{executivesummarybox}
```

### 機會方塊（青色/強調藍）
用於機會和正面發現。

```latex
\begin{opportunitybox}[成長機會]
亞太市場代表 150 億美元的機會，
年複合成長率為 22%。
\end{opportunitybox}
```

### 框架方塊
用於策略分析框架。

```latex
% SWOT 分析
\begin{swotbox}[SWOT 分析摘要]
內容...
\end{swotbox}

% 波特五力分析
\begin{porterbox}[波特五力分析]
內容...
\end{porterbox}
```

---

## 引用方塊

用於突顯重要統計數據或引用。

```latex
\begin{pullquote}
「人工智慧與醫療保健的融合代表到 2034 年
達 1990 億美元的機會。」
\end{pullquote}
```

---

## 統計方塊

用於突顯關鍵統計數據（以每行 3 個使用）。

```latex
\begin{center}
\statbox{\$452 億}{2024 年市場規模}
\statbox{15.3\%}{2024-2030 年年複合成長率}
\statbox{23\%}{市場領導者份額}
\end{center}
```

---

## 自訂命令

### 文字突顯
```latex
\highlight{重要文字}  % 藍色粗體
```

### 市場規模格式
```latex
\marketsize{452 億}   % 輸出：$452 億（綠色）
```

### 成長率格式
```latex
\growthrate{15.3}           % 輸出：15.3%（綠色）
```

### 風險指標
```latex
\riskhigh{}     % 輸出：高（紅色）
\riskmedium{}   % 輸出：中（橙色）
\risklow{}      % 輸出：低（綠色）
```

### 評級星星（1-5）
```latex
\rating{4}      % 輸出：★★★★☆
```

### 趨勢指標
```latex
\trendup{}      % 綠色上升三角形
\trenddown{}    % 紅色下降三角形
\trendflat{}    % 灰色右箭頭
```

---

## 表格格式

### 交替列的標準表格
```latex
\begin{table}[htbp]
\centering
\caption{各區域市場規模}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{區域} & \textbf{規模} & \textbf{份額} & \textbf{年複合成長率} \\
\midrule
北美 & \$182 億 & 40.3\% & 12.5\% \\
\rowcolor{tablealt} 歐洲 & \$121 億 & 26.8\% & 14.2\% \\
亞太地區 & \$105 億 & 23.2\% & 18.7\% \\
\rowcolor{tablealt} 其他地區 & \$44 億 & 9.7\% & 11.3\% \\
\midrule
\textbf{總計} & \textbf{\$452 億} & \textbf{100\%} & \textbf{15.3\%} \\
\bottomrule
\end{tabular}
\label{tab:regional}
\end{table}
```

### 含趨勢指標的表格
```latex
\begin{tabular}{@{}lrrl@{}}
\toprule
\textbf{公司} & \textbf{營收} & \textbf{份額} & \textbf{趨勢} \\
\midrule
公司 A & \$52 億 & 15.3\% & \trendup{} +12\% \\
公司 B & \$48 億 & 14.1\% & \trenddown{} -3\% \\
公司 C & \$42 億 & 12.4\% & \trendflat{} +1\% \\
\bottomrule
\end{tabular}
```

---

## 圖表格式

### 標準圖表
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{../figures/market_growth.png}
\caption{市場成長軌跡（2020-2030）}
\label{fig:growth}
\end{figure}
```

### 含來源標註的圖表
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{../figures/market_share.png}
\caption{市場份額分布（2024 年）}
\figuresource{公司年報、產業分析}
\label{fig:market_share}
\end{figure}
```

---

## 清單格式

### 項目符號清單
```latex
\begin{itemize}
    \item 第一項，自動藍色項目符號
    \item 第二項
    \item 第三項
\end{itemize}
```

### 編號清單
```latex
\begin{enumerate}
    \item 第一項，藍色數字
    \item 第二項
    \item 第三項
\end{enumerate}
```

### 巢狀清單
```latex
\begin{itemize}
    \item 主要重點
    \begin{itemize}
        \item 子項目 A
        \item 子項目 B
    \end{itemize}
    \item 另一個主要重點
\end{itemize}
```

---

## 標題頁

### 使用自訂標題命令
```latex
\makemarketreporttitle
    {市場標題}              % 報告標題
    {副標題}             % 副標題
    {../figures/cover.png}      % 主視覺圖像（留空則無圖像）
    {2025 年 1 月}              % 日期
    {市場情報團隊}  % 作者/編製方
```

### 手動標題頁
請參見範本以取得完整的手動標題頁程式碼。

---

## 附錄章節

```latex
\appendix

\chapter{方法論}

\appendixsection{資料來源}
出現在目錄中的內容...
```

---

## 常見模式

### 市場概況章節
```latex
\begin{marketdatabox}[市場概況]
\begin{itemize}
    \item \textbf{目前市場規模：} \marketsize{452 億}
    \item \textbf{預測規模（2030 年）：} \marketsize{987 億}
    \item \textbf{年複合成長率：} \growthrate{15.3}
    \item \textbf{最大區隔：} 企業（42% 份額）
    \item \textbf{成長最快區域：} 亞太地區（\growthrate{22.1} 年複合成長率）
\end{itemize}
\end{marketdatabox}
```

### 風險登記冊摘要
```latex
\begin{table}[htbp]
\centering
\caption{風險評估摘要}
\begin{tabular}{@{}llccl@{}}
\toprule
\textbf{風險} & \textbf{類別} & \textbf{機率} & \textbf{影響} & \textbf{評級} \\
\midrule
市場顛覆 & 市場 & 高 & 高 & \riskhigh{} \\
\rowcolor{tablealt} 法規變更 & 法規 & 中 & 高 & \riskhigh{} \\
新進入者 & 競爭 & 中 & 中 & \riskmedium{} \\
\rowcolor{tablealt} 技術過時 & 技術 & 低 & 高 & \riskmedium{} \\
匯率波動 & 財務 & 中 & 低 & \risklow{} \\
\bottomrule
\end{tabular}
\end{table}
```

### 競爭比較表
```latex
\begin{table}[htbp]
\centering
\caption{競爭比較}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{因素} & \textbf{公司 A} & \textbf{公司 B} & \textbf{公司 C} & \textbf{公司 D} \\
\midrule
市場份額 & \rating{5} & \rating{4} & \rating{3} & \rating{2} \\
\rowcolor{tablealt} 產品品質 & \rating{4} & \rating{5} & \rating{3} & \rating{4} \\
價格競爭力 & \rating{3} & \rating{3} & \rating{5} & \rating{4} \\
\rowcolor{tablealt} 創新 & \rating{5} & \rating{4} & \rating{2} & \rating{3} \\
客戶服務 & \rating{4} & \rating{4} & \rating{4} & \rating{5} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 故障排除

### 方塊溢位
如果方塊內容溢出頁面，分成多個方塊或使用分頁：
```latex
\newpage
\begin{keyinsightbox}[續...]
```

### 圖表位置
使用 `[htbp]` 進行彈性放置，或使用 `[H]`（需要 `float` 套件）進行精確放置：
```latex
\begin{figure}[H]  % 需要 \usepackage{float}
```

### 表格過寬
使用 `\resizebox` 或 `adjustbox`：
```latex
\resizebox{\textwidth}{!}{
\begin{tabular}{...}
...
\end{tabular}
}
```

### 顏色未顯示
確保 `xcolor` 套件以 `[table]` 選項載入（樣式檔案已包含）。

---

## 編譯

使用 XeLaTeX 以獲得最佳效果：
```bash
xelatex report.tex
bibtex report
xelatex report.tex
xelatex report.tex
```

或使用 latexmk：
```bash
latexmk -xelatex report.tex
```
