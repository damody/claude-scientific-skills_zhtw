````md
# 從 Bash 執行 MATLAB 和 GNU Octave 腳本

本文件展示從 Bash 環境使用 MATLAB（MathWorks）和 GNU Octave 執行 MATLAB 風格 `.m` 腳本的常見方法。涵蓋互動式使用、非互動式批次執行、傳遞參數、擷取輸出，以及自動化和 CI 的實用模式。

## 內容

- 需求
- 快速比較
- 從 Bash 執行 MATLAB 腳本
  - 互動模式
  - 非互動式執行腳本
  - 執行帶參數的函數
  - 執行單行指令
  - 工作目錄和路徑處理
  - 擷取輸出和退出碼
  - 常用的 MATLAB 腳本標誌
- 從 Bash 執行 Octave 腳本
  - 互動模式
  - 非互動式執行腳本
  - 執行帶參數的函數
  - 執行單行指令
  - 使 `.m` 檔案可執行（shebang）
  - 工作目錄和路徑處理
  - 擷取輸出和退出碼
  - 常用的 Octave 腳本標誌
- 跨平台相容性技巧（MATLAB + Octave）
- 範例：可攜式執行腳本
- 疑難排解

## 需求

### MATLAB
- 必須安裝 MATLAB。
- `matlab` 可執行檔必須在您的 PATH 中，或者您必須使用完整路徑引用它。
- 執行 MATLAB 需要有效的授權。

檢查：
```bash
matlab -help | head
````

### GNU Octave

* 必須安裝 Octave。
* `octave` 可執行檔必須在您的 PATH 中。

檢查：

```bash
octave --version
```

## 快速比較

| 任務                          | MATLAB                            | Octave                   |
| ----------------------------- | --------------------------------- | ------------------------ |
| 互動式 shell                  | `matlab`（預設 GUI）              | `octave`                 |
| 無頭執行（CI）                | `matlab -batch "cmd"`（建議）     | `octave --eval "cmd"`    |
| 執行腳本檔案                  | `matlab -batch "run('file.m')"`   | `octave --no-gui file.m` |
| 帶退出碼退出                  | `exit(n)`                         | `exit(n)`                |
| 使 `.m` 直接可執行            | 不常見                            | 常見（透過 shebang）     |

## 從 Bash 執行 MATLAB 腳本

### 1) 互動模式

啟動 MATLAB。根據您的平台和安裝方式，這可能會啟動 GUI。

```bash
matlab
```

對於僅終端機使用，建議使用 `-nodesktop`，可選 `-nosplash`：

```bash
matlab -nodesktop -nosplash
```

### 2) 非互動式執行腳本

建議的現代方法：`-batch`。它執行命令並在完成時退出。

使用 `run()` 執行腳本：

```bash
matlab -batch "run('myscript.m')"
```

如果腳本依賴從其目錄執行，請先設定工作目錄：

```bash
matlab -batch "cd('/path/to/project'); run('myscript.m')"
```

較舊的替代模式：`-r`（對自動化較不穩健，因為您必須確保 MATLAB 退出）：

```bash
matlab -nodisplay -nosplash -r "run('myscript.m'); exit"
```

### 3) 執行帶參數的函數

如果您的檔案定義了一個函數，直接呼叫它。建議使用 `-batch`：

```bash
matlab -batch "myfunc(123, 'abc')"
```

從 Bash 變數傳遞值：

```bash
matlab -batch "myfunc(${N}, '${NAME}')"
```

如果參數可能包含引號或空格，考慮撰寫一個小型 MATLAB 包裝函數來讀取環境變數。

### 4) 執行單行指令

```bash
matlab -batch "disp(2+2)"
```

多個語句：

```bash
matlab -batch "a=1; b=2; fprintf('%d\n', a+b)"
```

### 5) 工作目錄和路徑處理

常見選項：

* 啟動時變更目錄：

```bash
matlab -batch "cd('/path/to/project'); myfunc()"
```

* 將程式碼目錄加入 MATLAB 路徑：

```bash
matlab -batch "addpath('/path/to/lib'); myfunc()"
```

包含子資料夾：

```bash
matlab -batch "addpath(genpath('/path/to/project')); myfunc()"
```

### 6) 擷取輸出和退出碼

擷取 stdout/stderr：

```bash
matlab -batch "run('myscript.m')" > matlab.out 2>&1
```

檢查退出碼：

```bash
matlab -batch "run('myscript.m')"
echo $?
```

要明確讓流程失敗，在錯誤時使用 `exit(1)`。範例模式：

```matlab
try
  run('myscript.m');
catch ME
  disp(getReport(ME));
  exit(1);
end
exit(0);
```

執行它：

```bash
matlab -batch "try, run('myscript.m'); catch ME, disp(getReport(ME)); exit(1); end; exit(0);"
```

### 7) 常用的 MATLAB 腳本標誌

常用選項：

* `-batch "cmd"`：執行命令，返回程序退出碼，然後退出
* `-nodisplay`：無顯示（對無頭系統有用）
* `-nodesktop`：無桌面 GUI
* `-nosplash`：無啟動畫面
* `-r "cmd"`：執行命令；如果要終止必須包含 `exit`

確切的可用性因 MATLAB 版本而異，因此請使用 `matlab -help` 查看您的版本。

## 從 Bash 執行 GNU Octave 腳本

### 1) 互動模式

```bash
octave
```

較安靜：

```bash
octave --quiet
```

### 2) 非互動式執行腳本

執行檔案並退出：

```bash
octave --no-gui myscript.m
```

較安靜：

```bash
octave --quiet --no-gui myscript.m
```

某些環境使用：

```bash
octave --no-window-system myscript.m
```

### 3) 執行帶參數的函數

如果 `myfunc.m` 定義了函數 `myfunc`，透過 `--eval` 呼叫它：

```bash
octave --quiet --eval "myfunc(123, 'abc')"
```

如果您的函數不在 Octave 路徑中，先加入路徑：

```bash
octave --quiet --eval "addpath('/path/to/project'); myfunc()"
```

### 4) 執行單行指令

```bash
octave --quiet --eval "disp(2+2)"
```

多個語句：

```bash
octave --quiet --eval "a=1; b=2; printf('%d\n', a+b);"
```

### 5) 使 `.m` 檔案可執行（shebang）

這是 Octave 中常見的「獨立腳本」模式。

建立 `myscript.m`：

```matlab
#!/usr/bin/env octave
disp("Hello from Octave");
```

使其可執行：

```bash
chmod +x myscript.m
```

執行：

```bash
./myscript.m
```

如果需要標誌（quiet、no GUI），請改用包裝腳本，因為 shebang 行在不同平台上通常只支援有限的參數。

### 6) 工作目錄和路徑處理

執行前從 shell 變更目錄：

```bash
cd /path/to/project
octave --quiet --no-gui myscript.m
```

或在 Octave 內變更目錄：

```bash
octave --quiet --eval "cd('/path/to/project'); run('myscript.m');"
```

加入路徑：

```bash
octave --quiet --eval "addpath('/path/to/lib'); run('myscript.m');"
```

### 7) 擷取輸出和退出碼

擷取 stdout/stderr：

```bash
octave --quiet --no-gui myscript.m > octave.out 2>&1
```

退出碼：

```bash
octave --quiet --no-gui myscript.m
echo $?
```

要在錯誤時強制非零退出，包裝執行：

```matlab
try
  run('myscript.m');
catch err
  disp(err.message);
  exit(1);
end
exit(0);
```

執行它：

```bash
octave --quiet --eval "try, run('myscript.m'); catch err, disp(err.message); exit(1); end; exit(0);"
```

### 8) 常用的 Octave 腳本標誌

有用的選項：

* `--eval "cmd"`：執行命令字串
* `--quiet`：抑制啟動訊息
* `--no-gui`：停用 GUI
* `--no-window-system`：某些安裝上類似的無頭模式
* `--persist`：執行命令後保持 Octave 開啟（與批次行為相反）

檢查：

```bash
octave --help | head -n 50
```

## 跨平台相容性技巧（MATLAB 和 Octave）

1. 建議使用函數而非腳本進行自動化
   函數提供更清晰的參數傳遞和命名空間處理。

2. 如果需要可攜性，避免工具箱特定的呼叫
   許多 MATLAB 工具箱沒有 Octave 等效物。

3. 小心字串和引號
   MATLAB 和 Octave 都支援 `'單引號'`，而較新的 MATLAB 支援 `"雙引號"` 字串。為了最大相容性，除非您知道您的 Octave 版本以您需要的方式支援雙引號，否則建議使用單引號。

4. 使用 `fprintf` 或 `disp` 輸出
   對於 CI 日誌，保持輸出簡單且確定性。

5. 確保退出碼反映成功或失敗
   在兩個環境中，`exit(0)` 表示成功，`exit(1)` 表示失敗。

## 範例：可攜式 Bash 執行器

此腳本如果可用會先嘗試 MATLAB，否則使用 Octave。

建立 `run_mfile.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

FILE="${1:?用法: run_mfile.sh path/to/script_or_function.m}"
CMD="${2:-}"  # 可選命令覆寫

if command -v matlab >/dev/null 2>&1; then
  if [[ -n "$CMD" ]]; then
    matlab -batch "$CMD"
  else
    matlab -batch "run('${FILE}')"
  fi
elif command -v octave >/dev/null 2>&1; then
  if [[ -n "$CMD" ]]; then
    octave --quiet --no-gui --eval "$CMD"
  else
    octave --quiet --no-gui "$FILE"
  fi
else
  echo "在 PATH 中找不到 matlab 或 octave" >&2
  exit 127
fi
```

使其可執行：

```bash
chmod +x run_mfile.sh
```

執行：

```bash
./run_mfile.sh myscript.m
```

或執行函數呼叫：

```bash
./run_mfile.sh myfunc.m "myfunc(1, 'abc')"
```

## 疑難排解

### MATLAB: command not found

* 將 MATLAB 加入 PATH，或使用完整路徑呼叫它，例如：

```bash
/Applications/MATLAB_R202x?.app/bin/matlab -batch "disp('ok')"
```

### Octave: 伺服器上的 GUI 問題

* 使用 `--no-gui` 或 `--no-window-system`。

### 腳本依賴相對路徑

* 啟動前 `cd` 到腳本目錄，或在呼叫 `run()` 前在 MATLAB/Octave 內執行 `cd()`。

### 傳遞字串時的引號問題

* 避免在 `--eval` 或 `-batch` 中使用複雜引號。
* 當輸入複雜時，使用環境變數並在 MATLAB/Octave 內讀取它們。

### MATLAB 和 Octave 之間的不同行為

* 檢查不支援的函數或工具箱呼叫。
* 使用 `--eval` 或 `-batch` 執行最小重現步驟以隔離不相容性。
