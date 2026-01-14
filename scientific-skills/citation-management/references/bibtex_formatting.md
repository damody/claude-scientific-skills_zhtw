# BibTeX 格式化指南

BibTeX 條目類型、必填欄位、格式化慣例和最佳實務的完整指南。

## 概述

BibTeX 是 LaTeX 文件的標準參考書目格式。正確的格式化可確保：
- 正確的引用文獻呈現
- 一致的格式
- 與引用文獻樣式的相容性
- 無編譯錯誤

本指南涵蓋所有常見的條目類型和格式化規則。

## 條目類型

### @article - 期刊文章

**最常見的條目類型**，用於同行評審的期刊文章。

**必填欄位**：
- `author`：作者姓名
- `title`：文章標題
- `journal`：期刊名稱
- `year`：出版年份

**選用欄位**：
- `volume`：卷號
- `number`：期號
- `pages`：頁碼範圍
- `month`：出版月份
- `doi`：數位物件識別碼
- `url`：URL
- `note`：附註

**範本**：
```bibtex
@article{CitationKey2024,
  author  = {Last1, First1 and Last2, First2},
  title   = {Article Title Here},
  journal = {Journal Name},
  year    = {2024},
  volume  = {10},
  number  = {3},
  pages   = {123--145},
  doi     = {10.1234/journal.2024.123456},
  month   = jan
}
```

**範例**：
```bibtex
@article{Jumper2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title   = {Highly Accurate Protein Structure Prediction with {AlphaFold}},
  journal = {Nature},
  year    = {2021},
  volume  = {596},
  number  = {7873},
  pages   = {583--589},
  doi     = {10.1038/s41586-021-03819-2}
}
```

### @book - 書籍

**用於整本書籍**。

**必填欄位**：
- `author` 或 `editor`：作者或編輯
- `title`：書名
- `publisher`：出版商名稱
- `year`：出版年份

**選用欄位**：
- `volume`：卷號（如為多卷）
- `series`：叢書名稱
- `address`：出版商地點
- `edition`：版次
- `isbn`：ISBN
- `url`：URL

**範本**：
```bibtex
@book{CitationKey2024,
  author    = {Last, First},
  title     = {Book Title},
  publisher = {Publisher Name},
  year      = {2024},
  edition   = {3},
  address   = {City, Country},
  isbn      = {978-0-123-45678-9}
}
```

**範例**：
```bibtex
@book{Kumar2021,
  author    = {Kumar, Vinay and Abbas, Abul K. and Aster, Jon C.},
  title     = {Robbins and Cotran Pathologic Basis of Disease},
  publisher = {Elsevier},
  year      = {2021},
  edition   = {10},
  address   = {Philadelphia, PA},
  isbn      = {978-0-323-53113-9}
}
```

### @inproceedings - 會議論文

**用於會議論文集中的論文**。

**必填欄位**：
- `author`：作者姓名
- `title`：論文標題
- `booktitle`：會議/論文集名稱
- `year`：年份

**選用欄位**：
- `editor`：論文集編輯
- `volume`：卷號
- `series`：叢書名稱
- `pages`：頁碼範圍
- `address`：會議地點
- `month`：會議月份
- `organization`：主辦單位
- `publisher`：出版商
- `doi`：DOI

**範本**：
```bibtex
@inproceedings{CitationKey2024,
  author    = {Last, First},
  title     = {Paper Title},
  booktitle = {Proceedings of Conference Name},
  year      = {2024},
  pages     = {123--145},
  address   = {City, Country},
  month     = jun
}
```

**範例**：
```bibtex
@inproceedings{Vaswani2017,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  title     = {Attention is All You Need},
  booktitle = {Advances in Neural Information Processing Systems 30 (NeurIPS 2017)},
  year      = {2017},
  pages     = {5998--6008},
  address   = {Long Beach, CA}
}
```

**注意**：`@conference` 是 `@inproceedings` 的別名。

### @incollection - 書籍章節

**用於編輯書籍中的章節**。

**必填欄位**：
- `author`：章節作者
- `title`：章節標題
- `booktitle`：書名
- `publisher`：出版商名稱
- `year`：出版年份

**選用欄位**：
- `editor`：書籍編輯
- `volume`：卷號
- `series`：叢書名稱
- `type`：章節類型（例如「chapter」）
- `chapter`：章節號
- `pages`：頁碼範圍
- `address`：出版商地點
- `edition`：版次
- `month`：月份

**範本**：
```bibtex
@incollection{CitationKey2024,
  author    = {Last, First},
  title     = {Chapter Title},
  booktitle = {Book Title},
  editor    = {Editor, Last and Editor2, Last},
  publisher = {Publisher Name},
  year      = {2024},
  pages     = {123--145},
  chapter   = {5}
}
```

**範例**：
```bibtex
@incollection{Brown2020,
  author    = {Brown, Peter O. and Botstein, David},
  title     = {Exploring the New World of the Genome with {DNA} Microarrays},
  booktitle = {DNA Microarrays: A Molecular Cloning Manual},
  editor    = {Eisen, Michael B. and Brown, Patrick O.},
  publisher = {Cold Spring Harbor Laboratory Press},
  year      = {2020},
  pages     = {1--45},
  address   = {Cold Spring Harbor, NY}
}
```

### @phdthesis - 博士論文

**用於博士論文和學位論文**。

**必填欄位**：
- `author`：作者姓名
- `title`：論文標題
- `school`：機構
- `year`：年份

**選用欄位**：
- `type`：類型（例如「PhD dissertation」、「PhD thesis」）
- `address`：機構地點
- `month`：月份
- `url`：URL
- `note`：附註

**範本**：
```bibtex
@phdthesis{CitationKey2024,
  author = {Last, First},
  title  = {Dissertation Title},
  school = {University Name},
  year   = {2024},
  type   = {{PhD} dissertation},
  address = {City, State}
}
```

**範例**：
```bibtex
@phdthesis{Johnson2023,
  author  = {Johnson, Mary L.},
  title   = {Novel Approaches to Cancer Immunotherapy Using {CRISPR} Technology},
  school  = {Stanford University},
  year    = {2023},
  type    = {{PhD} dissertation},
  address = {Stanford, CA}
}
```

**注意**：`@mastersthesis` 類似，但用於碩士論文。

### @mastersthesis - 碩士論文

**用於碩士論文**。

**必填欄位**：
- `author`：作者姓名
- `title`：論文標題
- `school`：機構
- `year`：年份

**範本**：
```bibtex
@mastersthesis{CitationKey2024,
  author = {Last, First},
  title  = {Thesis Title},
  school = {University Name},
  year   = {2024}
}
```

### @misc - 其他

**用於不適合其他類別的項目**（預印本、資料集、軟體、網站等）。

**必填欄位**：
- `author`（如已知）
- `title`
- `year`

**選用欄位**：
- `howpublished`：儲存庫、網站、格式
- `url`：URL
- `doi`：DOI
- `note`：附加資訊
- `month`：月份

**預印本範本**：
```bibtex
@misc{CitationKey2024,
  author       = {Last, First},
  title        = {Preprint Title},
  year         = {2024},
  howpublished = {bioRxiv},
  doi          = {10.1101/2024.01.01.123456},
  note         = {Preprint}
}
```

**資料集範本**：
```bibtex
@misc{DatasetName2024,
  author       = {Last, First},
  title        = {Dataset Title},
  year         = {2024},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.123456},
  note         = {Version 1.2}
}
```

**軟體範本**：
```bibtex
@misc{SoftwareName2024,
  author       = {Last, First},
  title        = {Software Name},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/user/repo},
  note         = {Version 2.0}
}
```

### @techreport - 技術報告

**用於技術報告**。

**必填欄位**：
- `author`：作者姓名
- `title`：報告標題
- `institution`：機構
- `year`：年份

**選用欄位**：
- `type`：報告類型
- `number`：報告編號
- `address`：機構地點
- `month`：月份

**範本**：
```bibtex
@techreport{CitationKey2024,
  author      = {Last, First},
  title       = {Report Title},
  institution = {Institution Name},
  year        = {2024},
  type        = {Technical Report},
  number      = {TR-2024-01}
}
```

### @unpublished - 未發表作品

**用於未發表作品**（不是預印本 - 預印本使用 @misc）。

**必填欄位**：
- `author`：作者姓名
- `title`：作品標題
- `note`：描述

**選用欄位**：
- `month`：月份
- `year`：年份

**範本**：
```bibtex
@unpublished{CitationKey2024,
  author = {Last, First},
  title  = {Work Title},
  note   = {Unpublished manuscript},
  year   = {2024}
}
```

### @online/@electronic - 線上資源

**用於網頁和僅限線上的內容**。

**注意**：非標準 BibTeX，但許多參考書目套件支援（biblatex）。

**必填欄位**：
- `author` 或 `organization`
- `title`
- `url`
- `year`

**範本**：
```bibtex
@online{CitationKey2024,
  author = {{Organization Name}},
  title  = {Page Title},
  url    = {https://example.com/page},
  year   = {2024},
  note   = {Accessed: 2024-01-15}
}
```

## 格式化規則

### 引用鍵

**慣例**：`FirstAuthorYEARkeyword`

**範例**：
```bibtex
Smith2024protein
Doe2023machine
JohnsonWilliams2024cancer  % 多位作者，無空格
NatureEditorial2024        % 無作者，使用出版物
WHO2024guidelines          % 組織作者
```

**規則**：
- 英數字加上：`-`、`_`、`.`、`:`
- 無空格
- 區分大小寫
- 檔案內唯一
- 具描述性

**避免**：
- 特殊字元：`@`、`#`、`&`、`%`、`$`
- 空格：使用駝峰式命名或底線
- 以數字開頭：`2024Smith`（某些系統不允許）

### 作者姓名

**建議格式**：`Last, First Middle`

**單一作者**：
```bibtex
author = {Smith, John}
author = {Smith, John A.}
author = {Smith, John Andrew}
```

**多位作者** - 用 `and` 分隔：
```bibtex
author = {Smith, John and Doe, Jane}
author = {Smith, John A. and Doe, Jane M. and Johnson, Mary L.}
```

**多位作者**（10 位以上）：
```bibtex
author = {Smith, John and Doe, Jane and Johnson, Mary and others}
```

**特殊情況**：
```bibtex
% 後綴（Jr.、III 等）
author = {King, Jr., Martin Luther}

% 組織作為作者
author = {{World Health Organization}}
% 注意：雙大括號保持為單一實體

% 多姓氏
author = {Garc{\'i}a-Mart{\'i}nez, Jos{\'e}}

% 前綴（van、von、de 等）
author = {van der Waals, Johannes}
author = {de Broglie, Louis}
```

**錯誤格式**（不要使用）：
```bibtex
author = {Smith, J.; Doe, J.}  % 分號（錯誤）
author = {Smith, J., Doe, J.}  % 逗號（錯誤）
author = {Smith, J. & Doe, J.} % & 符號（錯誤）
author = {Smith J}             % 無逗號
```

### 標題大小寫

**使用大括號保護大小寫**：

```bibtex
% 專有名詞、縮寫、公式
title = {{AlphaFold}: Protein Structure Prediction}
title = {Machine Learning for {DNA} Sequencing}
title = {The {Ising} Model in Statistical Physics}
title = {{CRISPR-Cas9} Gene Editing Technology}
```

**原因**：引用文獻樣式可能會更改大小寫。大括號可保護。

**範例**：
```bibtex
% 正確
title = {Advances in {COVID-19} Treatment}
title = {Using {Python} for Data Analysis}
title = {The {AlphaFold} Protein Structure Database}

% 在標題格式樣式中會變小寫
title = {Advances in COVID-19 Treatment}  % covid-19
title = {Using Python for Data Analysis}  % python
```

**整個標題保護**（很少需要）：
```bibtex
title = {{This Entire Title Keeps Its Capitalization}}
```

### 頁碼範圍

**使用長破折號**（雙連字符 `--`）：

```bibtex
pages = {123--145}     % 正確
pages = {1234--1256}   % 正確
pages = {e0123456}     % 文章 ID（PLOS 等）
pages = {123}          % 單頁
```

**錯誤**：
```bibtex
pages = {123-145}      % 單連字符（不要使用）
pages = {pp. 123-145}  % 不需要「pp.」
pages = {123–145}      % Unicode 長破折號（可能導致問題）
```

### 月份名稱

**使用三字母縮寫**（不加引號）：

```bibtex
month = jan
month = feb
month = mar
month = apr
month = may
month = jun
month = jul
month = aug
month = sep
month = oct
month = nov
month = dec
```

**或數字**：
```bibtex
month = {1}   % 一月
month = {12}  % 十二月
```

**或大括號中的全名**：
```bibtex
month = {January}
```

**標準縮寫無需引號**，因為它們在 BibTeX 中已定義。

### 期刊名稱

**全名**（不縮寫）：

```bibtex
journal = {Nature}
journal = {Science}
journal = {Cell}
journal = {Proceedings of the National Academy of Sciences}
journal = {Journal of the American Chemical Society}
```

**參考書目樣式**會在需要時處理縮寫。

**避免手動縮寫**：
```bibtex
% 不要在 BibTeX 檔案中這樣做
journal = {Proc. Natl. Acad. Sci. U.S.A.}

% 改為這樣做
journal = {Proceedings of the National Academy of Sciences}
```

**例外**：如果樣式需要縮寫，使用完整的縮寫形式：
```bibtex
journal = {Proc. Natl. Acad. Sci. U.S.A.}  % 如果樣式要求
```

### DOI 格式化

**URL 格式**（建議）：

```bibtex
doi = {10.1038/s41586-021-03819-2}
```

**不是**：
```bibtex
doi = {https://doi.org/10.1038/s41586-021-03819-2}  % 不要包含 URL
doi = {doi:10.1038/s41586-021-03819-2}              % 不要包含前綴
```

**LaTeX** 會自動格式化為 URL。

**注意**：DOI 欄位後不要加句點！

### URL 格式化

```bibtex
url = {https://www.example.com/article}
```

**使用情況**：
- 無 DOI 時
- 網頁
- 補充材料

**不要重複**：
```bibtex
% 如果 DOI URL 與 url 相同，不要同時包含兩者
doi = {10.1038/nature12345}
url = {https://doi.org/10.1038/nature12345}  % 冗餘！
```

### 特殊字元

**重音和變音符號**：
```bibtex
author = {M{\"u}ller, Hans}        % ü
author = {Garc{\'i}a, Jos{\'e}}    % í, é
author = {Erd{\H{o}}s, Paul}       % ő
author = {Schr{\"o}dinger, Erwin}  % ö
```

**或使用 UTF-8**（需適當的 LaTeX 設定）：
```bibtex
author = {Müller, Hans}
author = {García, José}
```

**數學符號**：
```bibtex
title = {The $\alpha$-helix Structure}
title = {$\beta$-sheet Prediction}
```

**化學公式**：
```bibtex
title = {H$_2$O Molecular Dynamics}
% 或使用 chemformula 套件：
title = {\ce{H2O} Molecular Dynamics}
```

### 欄位順序

**建議順序**（為可讀性）：

```bibtex
@article{Key,
  author  = {},
  title   = {},
  journal = {},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {},
  url     = {},
  note    = {}
}
```

**規則**：
- 最重要的欄位優先
- 條目間保持一致
- 使用格式化工具標準化

## 最佳實務

### 1. 一致的格式化

全程使用相同格式：
- 作者姓名格式
- 標題大小寫
- 期刊名稱
- 引用鍵樣式

### 2. 必填欄位

始終包含：
- 條目類型的所有必填欄位
- 現代論文（2000 年以後）的 DOI
- 文章的卷號和頁碼
- 書籍的出版商

### 3. 保護大小寫

使用大括號：
- 專有名詞：`{AlphaFold}`
- 縮寫：`{DNA}`、`{CRISPR}`
- 公式：`{H2O}`
- 名稱：`{Python}`、`{R}`

### 4. 完整的作者列表

盡可能包含所有作者：
- 少於 10 位則包含所有
- 10 位以上使用「and others」
- 不要手動縮寫為「et al.」

### 5. 使用標準條目類型

選擇正確的條目類型：
- 期刊文章 → `@article`
- 書籍 → `@book`
- 會議論文 → `@inproceedings`
- 預印本 → `@misc`

### 6. 驗證語法

檢查：
- 大括號平衡
- 欄位後有逗號
- 引用鍵唯一
- 有效的條目類型

### 7. 使用格式化工具

使用自動化工具：
```bash
python scripts/format_bibtex.py references.bib
```

好處：
- 一致的格式化
- 捕捉語法錯誤
- 標準化欄位順序
- 修復常見問題

## 常見錯誤

### 1. 錯誤的作者分隔符

**錯誤**：
```bibtex
author = {Smith, J.; Doe, J.}    % 分號
author = {Smith, J., Doe, J.}    % 逗號
author = {Smith, J. & Doe, J.}   % & 符號
```

**正確**：
```bibtex
author = {Smith, John and Doe, Jane}
```

### 2. 遺失逗號

**錯誤**：
```bibtex
@article{Smith2024,
  author = {Smith, John}    % 遺失逗號！
  title = {Title}
}
```

**正確**：
```bibtex
@article{Smith2024,
  author = {Smith, John},   % 每個欄位後有逗號
  title = {Title}
}
```

### 3. 未保護的大小寫

**錯誤**：
```bibtex
title = {Machine Learning with Python}
% 「Python」在標題格式中會變成「python」
```

**正確**：
```bibtex
title = {Machine Learning with {Python}}
```

### 4. 頁碼範圍使用單連字符

**錯誤**：
```bibtex
pages = {123-145}   % 單連字符
```

**正確**：
```bibtex
pages = {123--145}  % 雙連字符（長破折號）
```

### 5. 頁碼中多餘的「pp.」

**錯誤**：
```bibtex
pages = {pp. 123--145}
```

**正確**：
```bibtex
pages = {123--145}
```

### 6. DOI 帶 URL 前綴

**錯誤**：
```bibtex
doi = {https://doi.org/10.1038/nature12345}
doi = {doi:10.1038/nature12345}
```

**正確**：
```bibtex
doi = {10.1038/nature12345}
```

## 完整參考書目範例

```bibtex
% 期刊文章
@article{Jumper2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  title   = {Highly Accurate Protein Structure Prediction with {AlphaFold}},
  journal = {Nature},
  year    = {2021},
  volume  = {596},
  number  = {7873},
  pages   = {583--589},
  doi     = {10.1038/s41586-021-03819-2}
}

% 書籍
@book{Kumar2021,
  author    = {Kumar, Vinay and Abbas, Abul K. and Aster, Jon C.},
  title     = {Robbins and Cotran Pathologic Basis of Disease},
  publisher = {Elsevier},
  year      = {2021},
  edition   = {10},
  address   = {Philadelphia, PA},
  isbn      = {978-0-323-53113-9}
}

% 會議論文
@inproceedings{Vaswani2017,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  title     = {Attention is All You Need},
  booktitle = {Advances in Neural Information Processing Systems 30 (NeurIPS 2017)},
  year      = {2017},
  pages     = {5998--6008}
}

% 書籍章節
@incollection{Brown2020,
  author    = {Brown, Peter O. and Botstein, David},
  title     = {Exploring the New World of the Genome with {DNA} Microarrays},
  booktitle = {DNA Microarrays: A Molecular Cloning Manual},
  editor    = {Eisen, Michael B. and Brown, Patrick O.},
  publisher = {Cold Spring Harbor Laboratory Press},
  year      = {2020},
  pages     = {1--45}
}

% 博士論文
@phdthesis{Johnson2023,
  author  = {Johnson, Mary L.},
  title   = {Novel Approaches to Cancer Immunotherapy},
  school  = {Stanford University},
  year    = {2023},
  type    = {{PhD} dissertation}
}

% 預印本
@misc{Zhang2024,
  author       = {Zhang, Yi and Chen, Li and Wang, Hui},
  title        = {Novel Therapeutic Targets in {Alzheimer}'s Disease},
  year         = {2024},
  howpublished = {bioRxiv},
  doi          = {10.1101/2024.01.001},
  note         = {Preprint}
}

% 資料集
@misc{AlphaFoldDB2021,
  author       = {{DeepMind} and {EMBL-EBI}},
  title        = {{AlphaFold} Protein Structure Database},
  year         = {2021},
  howpublished = {Database},
  url          = {https://alphafold.ebi.ac.uk/},
  doi          = {10.1093/nar/gkab1061}
}
```

## 總結

BibTeX 格式化要點：

✓ **選擇正確的條目類型**（@article、@book 等）
✓ **包含所有必填欄位**
✓ **多位作者使用 `and`**
✓ **用大括號保護大小寫**
✓ **頁碼範圍使用 `--`**
✓ **現代論文包含 DOI**
✓ **編譯前驗證語法**

使用格式化工具確保一致性：
```bash
python scripts/format_bibtex.py references.bib
```

正確格式化的 BibTeX 可確保在所有參考書目樣式中正確、一致地呈現引用文獻！
