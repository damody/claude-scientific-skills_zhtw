# 會議格式要求

跨學科主要學術會議的完整格式要求和投稿指南。

**最後更新**：2024

---

## 機器學習與人工智慧

### NeurIPS（Neural Information Processing Systems，神經資訊處理系統）

**會議類型**：頂級機器學習會議
**頻率**：每年（12 月）

**格式要求**：
- **頁數限制**：
  - 主文：8 頁（不含參考文獻）
  - 參考文獻：無限制
  - 附錄/補充材料：無限制（可選，審稿時酌情審閱）
- **格式**：雙欄
- **字體**：Times 或 Times New Roman，內文 10pt
- **行距**：單行距
- **邊界**：四邊各 1 英寸（2.54 cm）
- **欄距**：0.25 英寸（0.635 cm）
- **紙張大小**：US Letter（8.5 × 11 英寸）
- **匿名處理**：初次投稿**必須**（雙盲審查）
  - 移除作者姓名、機構
  - 匿名化自我引用（「Author et al.」→「Anonymous et al.」）
  - 移除暴露身份的致謝
- **引用**：方括號編號 [1], [2-4]
- **參考文獻**：任何一致的風格（常用編號參考文獻）
- **圖表**：
  - 高解析度（300+ dpi）
  - 建議使用色盲友好的調色板
  - 如需要可跨兩欄
- **表格**：清晰、出版尺寸下可讀
- **方程式**：如有引用則編號
- **LaTeX 類別**：`neurips_2024.sty`（每年更新）
- **補充材料**：
  - 強烈建議提供程式碼（GitHub，審稿用匿名倉庫）
  - 額外實驗、證明
  - 不計入頁數限制

**LaTeX 模板**：`assets/journals/neurips_article.tex`

**投稿注意事項**：
- 使用官方樣式檔（每年變更）
- 首頁須有論文 ID（投稿時自動生成）
- 包含「broader impact（更廣泛影響）」聲明（依年份而異）
- 需要可重複性檢查清單

**網站**：https://neurips.cc/

---

### ICML（International Conference on Machine Learning，國際機器學習會議）

**會議類型**：頂級機器學習會議
**頻率**：每年（7 月）

**格式要求**：
- **頁數限制**：
  - 主文：8 頁（不含參考文獻和附錄）
  - 參考文獻：無限制
  - 附錄：無限制（可選）
- **格式**：雙欄
- **字體**：Times，10pt
- **行距**：單行距
- **邊界**：四邊各 1 英寸
- **紙張大小**：US Letter
- **匿名處理**：**必須**（雙盲）
- **引用**：編號或作者-年份（風格一致）
- **圖表**：高解析度，建議色盲安全
- **LaTeX 類別**：`icml2024.sty`（每年更新）
- **補充材料**：強烈建議（程式碼、數據、附錄）

**LaTeX 模板**：`assets/journals/icml_article.tex`

**投稿注意事項**：
- 必須使用官方 ICML 樣式檔
- 可重複性檢查清單
- 如適用需包含倫理聲明

**網站**：https://icml.cc/

---

### ICLR（International Conference on Learning Representations，國際學習表徵會議）

**會議類型**：頂級深度學習會議
**頻率**：每年（4/5 月）

**格式要求**：
- **頁數限制**：
  - 主文：8 頁（不含參考文獻、附錄、倫理聲明）
  - 參考文獻：無限制
  - 附錄：無限制
- **格式**：雙欄
- **字體**：Times，10pt
- **匿名處理**：**必須**（雙盲）
- **引用**：編號 [1] 或作者-年份
- **LaTeX 類別**：`iclr2024_conference.sty`
- **補充材料**：建議提供程式碼和數據（匿名 GitHub）
- **Open Review**：決議後審稿和回覆公開

**LaTeX 模板**：`assets/journals/iclr_article.tex`

**獨特功能**：
- OpenReview 平台（透明審稿流程）
- 審稿期間作者-審稿人討論
- Camera-ready 可超過 8 頁

**網站**：https://iclr.cc/

---

### CVPR（Computer Vision and Pattern Recognition，電腦視覺與模式識別）

**會議類型**：頂級電腦視覺會議
**頻率**：每年（6 月）

**格式要求**：
- **頁數限制**：
  - 主文：8 頁（包含圖表，不含參考文獻）
  - 參考文獻：無限制（獨立章節）
- **格式**：雙欄
- **字體**：Times Roman，10pt
- **匿名處理**：**必須**（雙盲）
  - 如需要模糊圖片中的臉部
  - 匿名化可能暴露身份的數據集
- **紙張大小**：US Letter
- **引用**：編號 [1]
- **圖表**：高解析度，可為彩色
- **LaTeX 模板**：CVPR 官方模板（每年變更）
- **補充材料**：
  - 建議提供影片示範
  - 額外結果、程式碼
  - 所有補充檔案 100 MB 限制

**LaTeX 模板**：`assets/journals/cvpr_article.tex`

**網站**：https://cvpr.thecvf.com/

---

### AAAI（Association for the Advancement of Artificial Intelligence，美國人工智慧促進協會）

**會議類型**：主要人工智慧會議
**頻率**：每年（2 月）

**格式要求**：
- **頁數限制**：
  - 技術論文：7 頁（不含參考文獻）
  - 參考文獻：無限制
- **格式**：雙欄
- **字體**：Times Roman，10pt
- **匿名處理**：**必須**（雙盲）
- **紙張大小**：US Letter
- **引用**：接受各種風格（保持一致）
- **LaTeX 模板**：AAAI 官方樣式
- **補充材料**：可選附錄

**LaTeX 模板**：`assets/journals/aaai_article.tex`

**網站**：https://aaai.org/conference/aaai/

---

### IJCAI（International Joint Conference on Artificial Intelligence，國際人工智慧聯合會議）

**會議類型**：主要人工智慧會議
**頻率**：每年

**格式要求**：
- **頁數限制**：7 頁（不含參考文獻）
- **格式**：雙欄
- **字體**：Times，10pt
- **匿名處理**：**必須**
- **LaTeX 模板**：IJCAI 官方樣式

---

## 資訊科學

### ACM CHI（Human-Computer Interaction，人機互動）

**會議類型**：首要人機互動會議
**頻率**：每年（4/5 月）

**格式要求**：
- **頁數限制**：
  - 論文：10 頁（不含參考文獻）
  - Late-Breaking Work：4 頁
- **格式**：單欄 ACM 格式
- **字體**：依 ACM 模板而定
- **匿名處理**：Papers 類別**必須**
- **LaTeX 類別**：`acmart` 加 CHI proceedings 格式
- **引用**：ACM 風格（編號或作者-年份）
- **圖表**：高品質，考慮無障礙設計
- **無障礙設計**：建議為圖表提供替代文字

**LaTeX 模板**：`assets/journals/chi_article.tex`

**網站**：https://chi.acm.org/

---

### SIGKDD（Knowledge Discovery and Data Mining，知識發現與資料探勘）

**會議類型**：頂級資料探勘會議
**頻率**：每年（8 月）

**格式要求**：
- **頁數限制**：
  - 研究類別：9 頁（不含參考文獻）
  - 應用數據科學：9 頁
- **格式**：雙欄
- **LaTeX 類別**：`acmart`（sigconf 格式）
- **字體**：ACM 模板預設
- **匿名處理**：**必須**（雙盲）
- **引用**：ACM 編號風格
- **補充材料**：建議提供程式碼和數據

**LaTeX 模板**：`assets/journals/kdd_article.tex`

**網站**：https://kdd.org/

---

### EMNLP（Empirical Methods in Natural Language Processing，自然語言處理實證方法）

**會議類型**：頂級自然語言處理會議
**頻率**：每年（11/12 月）

**格式要求**：
- **頁數限制**：
  - 長論文：8 頁（+ 無限制參考文獻和附錄）
  - 短論文：4 頁（+ 無限制參考文獻）
- **格式**：雙欄
- **字體**：Times New Roman，11pt
- **匿名處理**：**必須**（雙盲）
  - 不包含作者姓名或機構
  - 自我引用應匿名化
- **紙張大小**：US Letter 或 A4
- **引用**：類似 ACL 的具名風格
- **LaTeX 模板**：ACL/EMNLP 官方樣式
- **補充材料**：附錄無限制，建議提供程式碼

**LaTeX 模板**：`assets/journals/emnlp_article.tex`

**網站**：https://www.emnlp.org/

---

### ACL（Association for Computational Linguistics，計算語言學協會）

**會議類型**：首要自然語言處理會議
**頻率**：每年（7 月）

**格式要求**：
- **頁數限制**：8 頁（長）、4 頁（短），不含參考文獻
- **格式**：雙欄
- **字體**：Times，11pt
- **匿名處理**：**必須**
- **LaTeX 模板**：ACL 官方樣式（acl.sty）

**LaTeX 模板**：`assets/journals/acl_article.tex`

---

### USENIX Security Symposium（USENIX 安全研討會）

**會議類型**：頂級安全會議
**頻率**：每年（8 月）

**格式要求**：
- **頁數限制**：
  - 論文：無嚴格限制（通常 15-20 頁含全部內容）
  - 偏好撰寫精良、簡潔的論文
- **格式**：雙欄
- **字體**：Times，10pt
- **匿名處理**：**必須**（雙盲）
- **LaTeX 模板**：USENIX 官方模板
- **引用**：編號
- **紙張大小**：US Letter

**LaTeX 模板**：`assets/journals/usenix_article.tex`

**網站**：https://www.usenix.org/conference/usenixsecurity

---

### SIGIR（Information Retrieval，資訊檢索）

**會議類型**：頂級資訊檢索會議
**頻率**：每年（7 月）

**格式要求**：
- **頁數限制**：
  - 完整論文：10 頁（不含參考文獻）
  - 短論文：4 頁（不含參考文獻）
- **格式**：單欄 ACM 格式
- **LaTeX 類別**：`acmart`（sigconf）
- **匿名處理**：**必須**
- **引用**：ACM 風格

**LaTeX 模板**：`assets/journals/sigir_article.tex`

---

## 生物學與生物資訊學

### ISMB（Intelligent Systems for Molecular Biology，分子生物學智慧系統）

**會議類型**：首要計算生物學會議
**頻率**：每年（7 月）

**格式要求**：
- **出版**：論文集發表於 *Bioinformatics* 期刊
- **頁數限制**：
  - 通常 7-8 頁，包含圖表和參考文獻
- **格式**：雙欄
- **字體**：Times，10pt
- **引用**：編號（類似 Bioinformatics 期刊的 Oxford 風格）
- **LaTeX 模板**：Oxford Bioinformatics 模板
- **匿名處理**：**不需要**（單盲）
- **圖表**：高解析度，可接受彩色
- **補充材料**：建議提供額外數據/方法

**LaTeX 模板**：`assets/journals/ismb_article.tex`

**網站**：https://www.iscb.org/ismb

---

### RECOMB（Research in Computational Molecular Biology，計算分子生物學研究）

**會議類型**：頂級計算生物學會議
**頻率**：每年（4/5 月）

**格式要求**：
- **出版**：論文集發表為 Springer LNCS（Lecture Notes in Computer Science）
- **頁數限制**：
  - 延伸摘要：12-15 頁（包含參考文獻）
- **格式**：單欄
- **字體**：依 Springer LNCS 模板
- **LaTeX 類別**：`llncs`（Springer）
- **引用**：編號或作者-年份
- **匿名處理**：**必須**（雙盲）
- **補充材料**：可提交附錄

**LaTeX 模板**：`assets/journals/recomb_article.tex`

**網站**：https://www.recomb.org/

---

### PSB（Pacific Symposium on Biocomputing，太平洋生物計算研討會）

**會議類型**：生物醫學資訊學會議
**頻率**：每年（1 月）

**格式要求**：
- **頁數限制**：12 頁，包含圖表和參考文獻
- **格式**：單欄
- **字體**：Times，11pt
- **邊界**：四邊各 1 英寸
- **引用**：編號
- **匿名處理**：**不需要**
- **圖表**：嵌入文中
- **LaTeX 模板**：PSB 官方模板

**LaTeX 模板**：`assets/journals/psb_article.tex`

**網站**：https://psb.stanford.edu/

---

## 工程

### IEEE International Conference on Robotics and Automation（ICRA，IEEE 國際機器人與自動化會議）

**格式要求**：
- **頁數限制**：8 頁（包含圖表和參考文獻）
- **格式**：雙欄
- **字體**：Times，10pt
- **LaTeX 類別**：IEEEtran
- **引用**：IEEE 風格 [1]
- **匿名處理**：初次投稿**必須**
- **影片**：建議提供可選影片投稿

**LaTeX 模板**：`assets/journals/icra_article.tex`

---

### IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS）

**格式**：與 ICRA 相同（IEEE 機器人模板）

---

### International Conference on Computer-Aided Design（ICCAD，國際電腦輔助設計會議）

**格式要求**：
- **頁數限制**：8 頁
- **格式**：雙欄
- **LaTeX 類別**：IEEE 模板
- **引用**：IEEE 風格

---

### Design Automation Conference（DAC，設計自動化會議）

**格式要求**：
- **頁數限制**：6 頁
- **格式**：雙欄
- **字體**：Times，10pt
- **LaTeX 類別**：ACM 或 IEEE 模板（查看年度指南）

---

## 跨學科

### AAAS Annual Meeting（AAAS 年會）

**會議類型**：廣泛科學會議
**格式**：依研討會而異（通常為延伸摘要）

---

## 快速參考表

| 會議 | 頁數 | 格式 | 盲審 | 引用 | 模板 |
|------------|-------|--------|-------|-----------|----------|
| **NeurIPS** | 8 + refs | 雙欄 | 雙盲 | [1] | `neurips_article.tex` |
| **ICML** | 8 + refs | 雙欄 | 雙盲 | [1] | `icml_article.tex` |
| **ICLR** | 8 + refs | 雙欄 | 雙盲 | [1] | `iclr_article.tex` |
| **CVPR** | 8 + refs | 雙欄 | 雙盲 | [1] | `cvpr_article.tex` |
| **AAAI** | 7 + refs | 雙欄 | 雙盲 | 各種 | `aaai_article.tex` |
| **CHI** | 10 + refs | 單欄 | 雙盲 | ACM | `chi_article.tex` |
| **SIGKDD** | 9 + refs | 雙欄 | 雙盲 | ACM [1] | `kdd_article.tex` |
| **EMNLP** | 8 + refs | 雙欄 | 雙盲 | 具名 | `emnlp_article.tex` |
| **ISMB** | 7-8 頁 | 雙欄 | 單盲 | [1] | `ismb_article.tex` |
| **RECOMB** | 12-15 頁 | 單欄 | 雙盲 | Springer | `recomb_article.tex` |

---

## 一般會議投稿指南

### 匿名化最佳實踐（雙盲審查）

**移除**：
- 標題頁的作者姓名、機構、電子郵件
- 致謝章節
- 暴露身份的資助資訊
- 任何明顯暴露身份的「我們之前的工作」引用

**匿名化**：
- 自我引用：「Smith et al. [5]」→「Anonymous et al. [5]」或「Prior work [5]」
- 機構特定細節：「our university」→「a large research university」
- 暴露身份的數據集名稱

**保持匿名**：
- 程式碼倉庫（審稿用匿名 GitHub）
- 補充材料
- 任何 URL 或連結

### 補充材料

**常見包含內容**：
- 原始碼（GitHub 倉庫、zip 檔案）
- 額外實驗結果
- 證明和推導
- 延伸相關工作
- 數據集描述
- 影片示範
- 互動演示

**最佳實踐**：
- 補充材料組織良好
- 從主文清楚引用補充材料
- 確保補充材料為盲審匿名化
- 檢查檔案大小限制（通常 50-100 MB）

### Camera-Ready 準備

接受後：
1. **取消匿名**：添加作者姓名、機構
2. **添加致謝**：資助、貢獻
3. **版權**：添加會議版權聲明
4. **格式**：遵循 camera-ready 特定指南
5. **頁數限制**：可能允許額外 1-2 頁（查看指南）
6. **PDF/A 合規**：某些會議要求 PDF/A 格式

### 無障礙設計考量

**適用於所有會議**：
- 使用色盲安全的調色板
- 確保足夠對比度
- 為圖表提供替代文字（如支援）
- 使用清晰、易讀的字體
- 避免僅依賴顏色的區分

---

## 常見錯誤避免

1. **錯誤樣式檔**：使用過時的會議樣式檔
2. **超出頁數限制**：圖表/表格超出限制
3. **字體大小調整**：更改字體以容納更多內容
4. **邊界調整**：修改邊界以獲得空間
5. **取消匿名化**：在盲審中意外暴露身份
6. **缺少參考文獻**：未引用相關先前工作
7. **低品質圖表**：像素化或難以辨認的圖表
8. **格式不一致**：不同章節使用不同風格

---

## 獲取官方模板

**找到官方模板的位置**：
1. **會議網站**：「Call for Papers」或「Author Instructions」
2. **GitHub**：許多會議在 GitHub 上托管模板
3. **Overleaf**：許多官方模板在 Overleaf 上可用
4. **CTAN**：LaTeX 類別檔通常在 CTAN 倉庫

**模板命名**：
- 會議通常每年更新模板
- 使用正確年份的模板（例如 `neurips_2024.sty`）
- 檢查「camera-ready」與「submission」版本

---

## 注意事項

1. **年度更新**：會議要求會變更；務必查看當年的 CFP
2. **截止日期類型**：
   - 摘要截止（通常在論文截止前 1 週）
   - 論文截止（嚴格，通常不延期）
   - 補充材料截止（可能在論文截止後幾天）
3. **時區**：注意截止日期時區（通常是 AOE - Anywhere on Earth）
4. **反駁**：許多會議有作者回覆/反駁期
5. **雙重投稿**：查看會議關於同時投稿的政策
6. **海報/口頭報告**：接受通常伴隨報告形式

## 會議層級（非正式）

**機器學習**：
- **第一層**：NeurIPS、ICML、ICLR
- **第二層**：AAAI、IJCAI、UAI

**電腦視覺**：
- **第一層**：CVPR、ICCV、ECCV

**自然語言處理**：
- **第一層**：ACL、EMNLP、NAACL

**生物資訊學**：
- **第一層**：RECOMB、ISMB
- **第二層**：PSB、WABI

（層級是非正式且依領域而定；非官方排名）

