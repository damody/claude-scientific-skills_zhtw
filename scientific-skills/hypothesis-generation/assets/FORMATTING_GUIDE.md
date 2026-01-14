# 假說生成報告 - 格式快速參考

## 概述

本指南提供使用假說生成 LaTeX 範本和樣式套件的快速參考。有關完整文件，請參閱 `SKILL.md`。

## 快速開始

```latex
% !TEX program = xelatex
\documentclass[11pt,letterpaper]{article}
\usepackage{hypothesis_generation}
\usepackage{natbib}

\title{Your Phenomenon Name}
\begin{document}
\maketitle
% Your content
\end{document}
```

**編譯：** 使用 XeLaTeX 或 LuaLaTeX 以獲得最佳結果
```bash
xelatex your_document.tex
bibtex your_document
xelatex your_document.tex
xelatex your_document.tex
```

## 配色方案參考

### 假說顏色
- **假說 1**：深藍色（RGB: 0, 102, 153）- 用於第一個假說
- **假說 2**：森林綠（RGB: 0, 128, 96）- 用於第二個假說
- **假說 3**：皇家紫（RGB: 102, 51, 153）- 用於第三個假說
- **假說 4**：青色（RGB: 0, 128, 128）- 用於第四個假說（如需要）
- **假說 5**：焦橙色（RGB: 204, 85, 0）- 用於第五個假說（如需要）

### 工具顏色
- **預測**：琥珀色（RGB: 255, 191, 0）- 用於可測試的預測
- **證據**：淺藍色（RGB: 102, 178, 204）- 用於支持證據
- **比較**：鋼灰色（RGB: 108, 117, 125）- 用於關鍵比較
- **限制**：珊瑚紅（RGB: 220, 53, 69）- 用於限制/挑戰

## 自訂框環境

### 1. 執行摘要框

```latex
\begin{summarybox}[Executive Summary]
  Content here
\end{summarybox}
```

**用於：** 文件開頭的高層概述

---

### 2. 假說框（5 種變體）

```latex
\begin{hypothesisbox1}[Hypothesis 1: Title]
  \textbf{Mechanistic Explanation:}
  [2-3 paragraphs explaining HOW and WHY]

  \textbf{Key Supporting Evidence:}
  \begin{itemize}
    \item Evidence point 1 \citep{ref1}
    \item Evidence point 2 \citep{ref2}
  \end{itemize}

  \textbf{Core Assumptions:}
  \begin{enumerate}
    \item Assumption 1
    \item Assumption 2
  \end{enumerate}
\end{hypothesisbox1}
```

**可用框：** `hypothesisbox1`、`hypothesisbox2`、`hypothesisbox3`、`hypothesisbox4`、`hypothesisbox5`

**用於：** 呈現每個競爭性假說及其機制、證據和假設

**4 頁主文的最佳實踐：**
- 將機制解釋保持在僅 1-2 個簡短段落（最多 6-10 句）
- 包含 2-3 個最重要的證據點並附引用
- 列出 1-2 個最關鍵的假設
- 確保每個假說真正獨特
- 所有詳細解釋放到附錄 A
- **在每個假說框之前使用 `\newpage` 以防止溢出**
- 每個完整的假說框應 ≤0.6 頁

---

### 3. 預測框

```latex
\begin{predictionbox}[Predictions: Hypothesis 1]
  \textbf{Prediction 1.1:} [Specific prediction]
  \begin{itemize}
    \item \textbf{Conditions:} When/where this applies
    \item \textbf{Expected Outcome:} Specific measurable result
    \item \textbf{Falsification:} What would disprove it
  \end{itemize}
\end{predictionbox}
```

**用於：** 從每個假說得出的可測試預測

**4 頁主文的最佳實踐：**
- 盡可能使預測具體且量化
- 清楚陳述預測應該成立的條件
- 始終指定證偽標準
- 在主文中每個假說僅包含 1-2 個最關鍵的預測
- 額外預測放到附錄

---

### 4. 證據框

```latex
\begin{evidencebox}[Supporting Evidence]
  Content discussing supporting evidence
\end{evidencebox}
```

**用於：** 突顯關鍵支持證據或文獻綜合

**最佳實踐：**
- 在主文中謹慎使用（詳細證據放在附錄 A）
- 為所有證據包含引用
- 聚焦於最有說服力的證據

---

### 5. 比較框

```latex
\begin{comparisonbox}[H1 vs. H2: Key Distinction]
  \textbf{Fundamental Difference:}
  [Description of core difference]

  \textbf{Discriminating Experiment:}
  [Description of experiment]

  \textbf{Outcome Interpretation:}
  \begin{itemize}
    \item \textbf{If [Result A]:} H1 supported
    \item \textbf{If [Result B]:} H2 supported
  \end{itemize}
\end{comparisonbox}
```

**用於：** 解釋如何區分競爭性假說

**最佳實踐：**
- 聚焦於基本機制差異
- 提出清晰、可行的區分實驗
- 指定具體的結果解釋
- 為所有主要假說配對建立比較

---

### 6. 限制框

```latex
\begin{limitationbox}[Limitations \& Challenges]
  Discussion of limitations
\end{limitationbox}
```

**用於：** 突顯重要的限制或挑戰

**最佳實踐：**
- 當限制特別重要時使用
- 誠實面對挑戰
- 建議如何解決限制

---

## 文件結構

### 主文（最多 4 頁 - 高度簡潔）

1. **執行摘要**（0.5-1 頁）
   - 使用 `summarybox`
   - 簡要現象概述
   - 每個假說用 1 句列出
   - 建議方法

2. **競爭性假說**（2-2.5 頁）
   - 使用 `hypothesisbox1`、`hypothesisbox2` 等
   - 每個假說一個框
   - 簡要機制解釋（1-2 段）+ 重要證據（2-3 點）+ 關鍵假設（1-2）
   - 目標：3-5 個假說
   - 保持高度簡潔 - 詳細內容放到附錄

3. **可測試的預測**（0.5-1 頁）
   - 為每個假說使用 `predictionbox`
   - 每個假說僅 1-2 個最關鍵的預測
   - 非常簡短 - 完整預測在附錄

4. **關鍵比較**（0.5-1 頁）
   - 僅為最高優先級的比較使用 `comparisonbox`
   - 展示如何區分最佳假說
   - 額外比較在附錄

**主文總計：最多 4 頁 - 對於放在這裡的內容要極度選擇性**

### 附錄（全面、詳細）

**附錄 A：全面文獻回顧**
- 詳細背景（大量引用）
- 當前理解
- 每個假說的證據（詳細）
- 相互矛盾的發現
- 知識差距
- **目標：40-60+ 引用**

**附錄 B：詳細實驗設計**
- 每個假說的完整方案
- 方法、對照、樣本量
- 統計方法
- 可行性評估
- 時間表和資源需求

**附錄 C：品質評估**
- 詳細評估表
- 優勢和劣勢分析
- 比較評分
- 建議

**附錄 D：補充證據**
- 類比機制
- 初步資料
- 理論框架
- 歷史背景

**參考文獻**
- **目標：總共 50+ 參考文獻**

## 引用最佳實踐

### 在主文中
- 引用 15-20 篇關鍵論文
- 使用 `\citep{author2023}` 進行括號引用
- 使用 `\citet{author2023}` 進行文字引用
- 聚焦於最重要/最新的證據

### 在附錄中
- 總共引用 40-60+ 篇論文
- 全面涵蓋相關文獻
- 包含評論、原始研究、理論論文
- 為每個主張和證據引用

### 引用密度指南
- 主要假說框：每個框 2-3 個引用（僅最重要的）
- 主文總計：最多 10-15 個引用（保持簡潔）
- 附錄 A 文獻部分：每個子部分 8-15 個引用
- 實驗設計：2-5 個引用用於方法/先例
- 品質評估：根據評估標準需要的引用
- 全文總計：50+ 引用（絕大多數在附錄）

## 表格

### 專業表格格式

```latex
\begin{hypotable}{Caption}
\begin{tabular}{|l|l|l|}
\hline
\tableheadercolor
\textcolor{white}{\textbf{Header 1}} & \textcolor{white}{\textbf{Header 2}} \\
\hline
Data row 1 & Data \\
\hline
\tablerowcolor  % Alternating gray background
Data row 2 & Data \\
\hline
\end{tabular}
\caption{Your caption}
\end{hypotable}
```

**最佳實踐：**
- 對表頭行使用 `\tableheadercolor`
- 對超過 3 行的表格交替使用 `\tablerowcolor`
- 保持表格可讀（不要太寬）
- 用於品質評估、比較

## 常見格式模式

### 假說部分模式

```latex
% Use \newpage before hypothesis box to prevent overflow
\newpage
\subsection*{Hypothesis N: [Concise Title]}

\begin{hypothesisboxN}[Hypothesis N: [Title]]

\textbf{Mechanistic Explanation:}

[1-2 brief paragraphs of explanation - 6-10 sentences max]

\vspace{0.3cm}

\textbf{Key Supporting Evidence:}
\begin{itemize}
  \item [Evidence 1] \citep{ref1}
  \item [Evidence 2] \citep{ref2}
  \item [Evidence 3] \citep{ref3}
\end{itemize}

\vspace{0.3cm}

\textbf{Core Assumptions:}
\begin{enumerate}
  \item [Assumption 1]
  \item [Assumption 2]
\end{enumerate}

\end{hypothesisboxN}

\vspace{0.5cm}
```

**注意：** 假說框之前的 `\newpage` 確保它從新頁面開始，防止溢出。當框包含大量內容時這特別重要。

### 預測部分模式

```latex
\subsection*{Predictions from Hypothesis N}

\begin{predictionbox}[Predictions: Hypothesis N]

\textbf{Prediction N.1:} [Statement]
\begin{itemize}
  \item \textbf{Conditions:} [Conditions]
  \item \textbf{Expected Outcome:} [Outcome]
  \item \textbf{Falsification:} [Falsification]
\end{itemize}

\vspace{0.2cm}

\textbf{Prediction N.2:} [Statement]
[... continue ...]

\end{predictionbox}
```

### 比較部分模式

```latex
\subsection*{Distinguishing Hypothesis X vs. Hypothesis Y}

\begin{comparisonbox}[HX vs. HY: Key Distinction]

\textbf{Fundamental Difference:}

[Description of core difference]

\vspace{0.3cm}

\textbf{Discriminating Experiment:}

[Experiment description]

\vspace{0.3cm}

\textbf{Outcome Interpretation:}
\begin{itemize}
  \item \textbf{If [Result A]:} HX supported
  \item \textbf{If [Result B]:} HY supported
  \item \textbf{If [Result C]:} Both/neither supported
\end{itemize}

\end{comparisonbox}
```

## 間距和版面配置

### 垂直間距
- `\vspace{0.3cm}` - 框內元素之間
- `\vspace{0.5cm}` - 主要部分或框之間
- `\vspace{1cm}` - 標題之後、主要內容之前

### 分頁符和溢出防止

**關鍵：防止內容溢出**

LaTeX 框（tcolorbox 環境）不會自動跨頁分割。超過剩餘頁面空間的內容將溢出並導致格式問題。請遵循這些指南：

1. **在長框之前使用策略性分頁符：**
```latex
\newpage  % Start on fresh page if box will be long
\begin{hypothesisbox1}[Hypothesis 1: Title]
  % Substantial content here
\end{hypothesisbox1}
```

2. **監控框內容長度：**
   - 每個假說框最多應為 ≤0.7 頁
   - 如果機制解釋 + 證據 + 假設超過約 0.6 頁，內容太長
   - 解決方案：將詳細內容移至附錄，僅保留主文框中的要點

3. **何時使用 `\newpage`：**
   - 在具有 >3 個子部分或 >15 行內容的任何假說框之前
   - 在具有大量實驗描述的比較框之前
   - 在主要附錄部分之間
   - 如果當前頁面在開始新框之前剩餘不到 0.6 頁

4. **主文內容長度指南：**
   - 執行摘要框：最多 0.5-0.8 頁
   - 每個假說框：最多 0.4-0.6 頁
   - 每個預測框：最多 0.3-0.5 頁
   - 每個比較框：最多 0.4-0.6 頁

5. **分割長內容：**
   ```latex
   % GOOD: Concise main text with page break
   \newpage
   \begin{hypothesisbox1}[Hypothesis 1: Brief Title]
   \textbf{Mechanistic Explanation:}
   Brief overview in 1-2 paragraphs (6-10 sentences).

   \textbf{Key Supporting Evidence:}
   \begin{itemize}
     \item Evidence 1 \citep{ref1}
     \item Evidence 2 \citep{ref2}
   \end{itemize}

   \textbf{Core Assumptions:}
   \begin{enumerate}
     \item Assumption 1
   \end{enumerate}

   See Appendix A for detailed mechanism and comprehensive evidence.
   \end{hypothesisbox1}
   ```

   ```latex
   % BAD: Overly long content that will overflow
   \begin{hypothesisbox1}[Hypothesis 1]
   \subsection{Very Long Section}
   Multiple paragraphs...
   \subsection{Another Long Section}
   More paragraphs...
   \subsection{Even More Content}
   [Content continues beyond page boundary → OVERFLOW!]
   \end{hypothesisbox1}
   ```

6. **分頁符命令：**
   - `\newpage` - 強制新頁面（建議在長框之前使用）
   - `\clearpage` - 強制新頁面並清除浮動物件（在附錄之前使用）

### 部分間距
已由樣式套件處理，但您可以調整：
```latex
\vspace{0.5cm}  % Add extra space if needed
```

## 故障排除

### 常見問題

**問題：「找不到 hypothesis_generation.sty 檔案」**
- 解決方案：確保 .sty 檔案與您的 .tex 檔案在同一目錄中，或在您的 LaTeX 路徑中

**問題：框沒有顏色**
- 解決方案：使用 XeLaTeX 或 LuaLaTeX 編譯，而不是 pdfLaTeX
- 命令：`xelatex yourfile.tex`

**問題：引用顯示為 [?]**
- 解決方案：在第一次 xelatex 編譯後運行 bibtex
```bash
xelatex yourfile.tex
bibtex yourfile
xelatex yourfile.tex
xelatex yourfile.tex
```

**問題：找不到字型**
- 解決方案：如果未安裝自訂字型，請註解掉 .sty 檔案中的字型行
- 要註解的行：`\setmainfont{...}` 和 `\setsansfont{...}`

**問題：框標題與內容重疊**
- 解決方案：在標題後添加更多垂直間距 `\vspace{0.3cm}`

**問題：表格太寬**
- 解決方案：在 tabular 之前使用 `\small` 或 `\footnotesize`，或使用 `p{width}` 列規格

**問題：內容溢出頁面**
- **原因：** 框（tcolorbox 環境）太長無法放入剩餘頁面空間
- **解決方案 1：** 在框之前添加 `\newpage` 以在新頁面開始
- **解決方案 2：** 減少框內容 - 將詳細資訊移至附錄
- **解決方案 3：** 將內容分成多個較小的框
- **預防：** 將每個假說框保持在最多 0.4-0.6 頁；在具有大量內容的框之前自由使用 `\newpage`

**問題：主文超過 4 頁**
- **原因：** 框包含太多詳細資訊
- **解決方案：** 積極地將內容移至附錄 - 主文框應該只包含：
  - 簡要機制概述（1-2 段）
  - 2-3 個關鍵證據要點
  - 1-2 個核心假設
- 所有詳細解釋、額外證據和全面討論都屬於附錄 A

### 套件需求

確保安裝了這些套件：
- `tcolorbox`（帶 `most` 選項）
- `xcolor`
- `fontspec`（用於 XeLaTeX/LuaLaTeX）
- `fancyhdr`
- `titlesec`
- `enumitem`
- `booktabs`
- `natbib`

安裝缺失的套件：
```bash
# For TeX Live
tlmgr install tcolorbox xcolor fontspec fancyhdr titlesec enumitem booktabs natbib

# For MiKTeX (Windows)
# Use MiKTeX Package Manager GUI
```

## 風格一致性提示

1. **顏色使用**
   - 在整個文件中始終對每個假說使用相同的顏色
   - H1 = 藍色，H2 = 綠色，H3 = 紫色，等等。
   - 不要為同一假說混用顏色

2. **框使用**
   - 主文：假說框、預測框、比較框
   - 附錄：根據需要可以使用證據框、限制框
   - 不要過度使用框 - 保留給關鍵內容

3. **引用風格**
   - 全文一致的引用格式
   - 大多數引用使用 `\citep{}`
   - 組合多個引用：`\citep{ref1, ref2, ref3}`

4. **假說編號**
   - 一致地編號假說（H1、H2、H3 等）
   - 在預測中使用相同編號（P1.1、P1.2 用於 H1）
   - 在比較中使用相同編號（H1 vs. H2）

5. **語言**
   - 精確和具體
   - 避免模糊語言（「可能」、「或許」、「大概」）
   - 盡可能使用主動語態
   - 在可行時使預測量化

## 快速檢查清單

在完成文件之前：

- [ ] 標題頁有現象名稱
- [ ] **主文最多 4 頁**
- [ ] 執行摘要簡潔（0.5-1 頁）
- [ ] 每個假說在其自己的彩色框中
- [ ] 呈現 3-5 個假說（不要更多）
- [ ] 每個假說有簡要機制解釋（1-2 段）
- [ ] 每個假說有 2-3 個最重要的證據點並附引用
- [ ] 每個假說有 1-2 個最關鍵的假設
- [ ] 預測框每個假說有 1-2 個關鍵預測
- [ ] 主文中有優先比較框（其他在附錄）
- [ ] 已識別優先實驗
- [ ] **在長框之前使用分頁符（`\newpage`）以防止溢出**
- [ ] **沒有內容溢出頁面邊界（仔細檢查 PDF）**
- [ ] **每個假說框 ≤0.6 頁（如果更長，將詳細資訊移至附錄）**
- [ ] 附錄 A 有包含詳細證據的全面文獻回顧
- [ ] 附錄 B 有詳細實驗方案
- [ ] 附錄 C 有品質評估表
- [ ] 附錄 D 有補充證據
- [ ] 主文中 10-15 個引用（選擇性）
- [ ] 全文總共 50+ 引用
- [ ] 所有框使用正確的顏色
- [ ] 文件編譯無錯誤
- [ ] 參考文獻格式正確
- [ ] **已視覺檢查編譯的 PDF 是否有溢出問題**

## 最小範例文件

```latex
% !TEX program = xelatex
\documentclass[11pt,letterpaper]{article}
\usepackage{hypothesis_generation}
\usepackage{natbib}

\title{Role of X in Y}

\begin{document}
\maketitle

\section*{Executive Summary}
\begin{summarybox}[Executive Summary]
Brief overview of phenomenon and hypotheses.
\end{summarybox}

\section{Competing Hypotheses}

% Use \newpage before each hypothesis box to prevent overflow
\newpage
\subsection*{Hypothesis 1: Title}
\begin{hypothesisbox1}[Hypothesis 1: Title]
\textbf{Mechanistic Explanation:}
Brief explanation in 1-2 paragraphs.

\textbf{Key Supporting Evidence:}
\begin{itemize}
  \item Evidence point \citep{ref1}
\end{itemize}
\end{hypothesisbox1}

\newpage
\subsection*{Hypothesis 2: Title}
\begin{hypothesisbox2}[Hypothesis 2: Title]
\textbf{Mechanistic Explanation:}
Brief explanation in 1-2 paragraphs.

\textbf{Key Supporting Evidence:}
\begin{itemize}
  \item Evidence point \citep{ref2}
\end{itemize}
\end{hypothesisbox2}

\section{Testable Predictions}

\subsection*{Predictions from Hypothesis 1}
\begin{predictionbox}[Predictions: Hypothesis 1]
Predictions here.
\end{predictionbox}

\section{Critical Comparisons}

\subsection*{H1 vs. H2}
\begin{comparisonbox}[H1 vs. H2]
Comparison here.
\end{comparisonbox}

% Force new page before appendices
\appendix
\newpage
\appendixsection{Appendix A: Literature Review}
Detailed literature review here.

\newpage
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

**關鍵要點：**
- 在每個假說框之前使用 `\newpage` 以確保它們從新頁面開始
- 這可以防止內容溢出問題
- 主文框保持簡潔（1-2 段 + 要點）
- 詳細內容放到附錄

## 額外資源

- 請參閱 `hypothesis_report_template.tex` 以獲取完整的註解範本
- 請參閱 `SKILL.md` 以獲取工作流程和方法論指導
- 請參閱 `references/hypothesis_quality_criteria.md` 以獲取評估框架
- 請參閱 `references/experimental_design_patterns.md` 以獲取設計指導
- 請參閱 treatment-plans 技能以獲取額外的 LaTeX 樣式範例

