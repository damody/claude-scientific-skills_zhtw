# 專業治療計畫樣式 - 快速參考

## 檔案位置
`medical_treatment_plan.sty` - 位於 assets 目錄中

## 快速開始

```latex
% !TEX program = xelatex
\documentclass[11pt,letterpaper]{article}
\usepackage{medical_treatment_plan}
\usepackage{natbib}

\begin{document}
\maketitle
% Your content
\end{document}
```

## 自訂框環境

### 1. 資訊框（藍色）- 一般資訊
```latex
\begin{infobox}[Title]
  Content
\end{infobox}
```
**用於：** 臨床評估、監測排程、滴定方案

### 2. 警告框（黃色/紅色）- 關鍵警報
```latex
\begin{warningbox}[Title]
  Critical information
\end{warningbox}
```
**用於：** 安全協定、決策點、禁忌症

### 3. 目標框（綠色）- 治療目標
```latex
\begin{goalbox}[Title]
  Goals and targets
\end{goalbox}
```
**用於：** SMART 目標、目標結果、成功指標

### 4. 關鍵點框（淺藍色）- 重點
```latex
\begin{keybox}[Title]
  Important highlights
\end{keybox}
```
**用於：** 執行摘要、關鍵要點、重要建議

### 5. 緊急框（紅色）- 緊急資訊
```latex
\begin{emergencybox}
  Emergency contacts
\end{emergencybox}
```
**用於：** 緊急聯絡人、緊急協定

### 6. 病患資訊框（白色/藍色）- 人口統計
```latex
\begin{patientinfo}
  Patient information
\end{patientinfo}
```
**用於：** 病患人口統計和基線資料

## 專業表格

```latex
\begin{medtable}{Caption}
\begin{tabular}{|l|l|l|}
\hline
\tableheadercolor
\textcolor{white}{\textbf{Header 1}} & \textcolor{white}{\textbf{Header 2}} \\
\hline
Data row 1 \\
\hline
\tablerowcolor  % Alternating gray
Data row 2 \\
\hline
\end{tabular}
\caption{Table caption}
\end{medtable}
```

## 色彩方案

- **主要藍色**（0, 102, 153）：標題、標題
- **次要藍色**（102, 178, 204）：淺色背景
- **強調藍色**（0, 153, 204）：連結、重點
- **成功綠色**（0, 153, 76）：目標
- **警告紅色**（204, 0, 0）：警告

## 編譯

```bash
xelatex document.tex
bibtex document
xelatex document.tex
xelatex document.tex
```

## 最佳實踐

1. **將框類型與目的匹配：** 綠色用於目標，紅色/黃色用於警告
2. **不要過度使用框：** 僅保留給重要資訊
3. **保持色彩一致性：** 遵循定義的方案
4. **使用留白：** 在主要區段之間添加 `\vspace{0.5cm}`
5. **表格交替列：** 使用 `\tablerowcolor` 提高可讀性

## 安裝

**選項 1：** 複製 `assets/medical_treatment_plan.sty` 到您的文件目錄

**選項 2：** 安裝到使用者 TeX 目錄
```bash
mkdir -p ~/texmf/tex/latex/medical_treatment_plan
cp assets/medical_treatment_plan.sty ~/texmf/tex/latex/medical_treatment_plan/
texhash ~/texmf
```

## 必要套件
由樣式自動載入：
- tcolorbox、tikz、xcolor
- fancyhdr、titlesec、enumitem
- booktabs、longtable、array、colortbl
- hyperref、natbib、fontspec

## 範例結構

```latex
\maketitle

\section*{Patient Information}
\begin{patientinfo}
  Demographics
\end{patientinfo}

\section{Executive Summary}
\begin{keybox}[Plan Overview]
  Key highlights
\end{keybox}

\section{Treatment Goals}
\begin{goalbox}[SMART Goals]
  Goals list
\end{goalbox}

\section{Medication Plan}
\begin{infobox}[Dosing]
  Instructions
\end{infobox}

\begin{warningbox}[Safety]
  Warnings
\end{warningbox}

\section{Emergency}
\begin{emergencybox}
  Contacts
\end{emergencybox}
```

## 疑難排解

**缺少套件：**
```bash
sudo tlmgr install tcolorbox tikz pgf
```

**特殊字元未顯示：**
- 使用 XeLaTeX 而非 PDFLaTeX
- 或使用 LaTeX 命令：`$\checkmark$`、`$\geq$`

**標題警告：**
- 樣式檔案中已設定為 22pt
- 如需要可調整

---

有關完整文件，請參閱 SKILL.md 中的「專業文件樣式」區段
