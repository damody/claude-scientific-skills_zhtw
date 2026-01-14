# Python 整合參考

## 目錄
1. [從 MATLAB 呼叫 Python](#從-matlab-呼叫-python)
2. [資料類型轉換](#資料類型轉換)
3. [使用 Python 物件](#使用-python-物件)
4. [從 Python 呼叫 MATLAB](#從-python-呼叫-matlab)
5. [常見工作流程](#常見工作流程)

## 從 MATLAB 呼叫 Python

### 設定

```matlab
% 檢查 Python 設定
pyenv

% 設定 Python 版本（在呼叫任何 Python 之前）
pyenv('Version', '/usr/bin/python3');
pyenv('Version', '3.10');

% 檢查 Python 是否可用
pe = pyenv;
disp(pe.Version);
disp(pe.Executable);
```

### 基本 Python 呼叫

```matlab
% 使用 py. 前綴呼叫內建函數
result = py.len([1, 2, 3, 4]);  % 4
result = py.sum([1, 2, 3, 4]);  % 10
result = py.max([1, 2, 3, 4]);  % 4
result = py.abs(-5);            % 5

% 建立 Python 物件
pyList = py.list({1, 2, 3});
pyDict = py.dict(pyargs('a', 1, 'b', 2));
pySet = py.set({1, 2, 3});
pyTuple = py.tuple({1, 2, 3});

% 呼叫模組函數
result = py.math.sqrt(16);
result = py.os.getcwd();
wrapped = py.textwrap.wrap('這是一個長字串');
```

### 匯入與使用模組

```matlab
% 匯入模組
np = py.importlib.import_module('numpy');
pd = py.importlib.import_module('pandas');

% 使用模組
arr = np.array({1, 2, 3, 4, 5});
result = np.mean(arr);

% 替代方案：直接 py. 語法
arr = py.numpy.array({1, 2, 3, 4, 5});
result = py.numpy.mean(arr);
```

### 執行 Python 程式碼

```matlab
% 執行 Python 語句
pyrun("x = 5")
pyrun("y = x * 2")
result = pyrun("z = y + 1", "z");

% 執行 Python 檔案
pyrunfile("script.py");
result = pyrunfile("script.py", "output_variable");

% 帶輸入變數執行
x = 10;
result = pyrun("y = x * 2", "y", x=x);
```

### 關鍵字參數

```matlab
% 使用 pyargs 傳遞關鍵字參數
result = py.sorted({3, 1, 4, 1, 5}, pyargs('reverse', true));

% 多個關鍵字參數
df = py.pandas.DataFrame(pyargs( ...
    'data', py.dict(pyargs('A', {1, 2, 3}, 'B', {4, 5, 6})), ...
    'index', {'x', 'y', 'z'}));
```

## 資料類型轉換

### MATLAB 到 Python

| MATLAB 類型 | Python 類型 |
|-------------|-------------|
| double, single | float |
| int8, int16, int32, int64 | int |
| uint8, uint16, uint32, uint64 | int |
| logical | bool |
| char, string | str |
| cell array（儲存格陣列） | list |
| struct（結構體） | dict |
| numeric array（數值陣列） | numpy.ndarray（如果 numpy 可用） |

```matlab
% 自動轉換範例
py.print(3.14);         % float
py.print(int32(42));    % int
py.print(true);         % bool (True)
py.print("hello");      % str
py.print({'a', 'b'});   % list

% 明確轉換為 Python 類型
pyInt = py.int(42);
pyFloat = py.float(3.14);
pyStr = py.str('hello');
pyList = py.list({1, 2, 3});
pyDict = py.dict(pyargs('key', 'value'));
```

### Python 到 MATLAB

```matlab
% 轉換 Python 類型到 MATLAB
matlabDouble = double(py.float(3.14));
matlabInt = int64(py.int(42));
matlabChar = char(py.str('hello'));
matlabString = string(py.str('hello'));
matlabCell = cell(py.list({1, 2, 3}));

% 轉換 numpy 陣列
pyArr = py.numpy.array({1, 2, 3, 4, 5});
matlabArr = double(pyArr);

% 轉換 pandas DataFrame 到 MATLAB 表格
pyDf = py.pandas.read_csv('data.csv');
matlabTable = table(pyDf);  % 需要 pandas2table 或類似工具

% 手動 DataFrame 轉換
colNames = cell(pyDf.columns.tolist());
data = cell(pyDf.values.tolist());
T = cell2table(data, 'VariableNames', colNames);
```

### 陣列轉換

```matlab
% MATLAB 陣列到 numpy
matlabArr = [1 2 3; 4 5 6];
pyArr = py.numpy.array(matlabArr);

% numpy 到 MATLAB
pyArr = py.numpy.random.rand(int64(3), int64(4));
matlabArr = double(pyArr);

% 注意：numpy 使用列優先（C）順序，MATLAB 使用欄優先（Fortran）順序
% 可能需要轉置以取得正確的排列
```

## 使用 Python 物件

### 物件方法與屬性

```matlab
% 呼叫方法
pyList = py.list({3, 1, 4, 1, 5});
pyList.append(9);
pyList.sort();

% 存取屬性
pyStr = py.str('hello world');
upper = pyStr.upper();
words = pyStr.split();

% 檢查屬性
methods(pyStr)          % 列出方法
fieldnames(pyDict)      % 列出鍵
```

### 迭代 Python 物件

```matlab
% 迭代 Python 列表
pyList = py.list({1, 2, 3, 4, 5});
for item = py.list(pyList)
    disp(item{1});
end

% 轉換為儲存格並迭代
items = cell(pyList);
for i = 1:length(items)
    disp(items{i});
end

% 迭代字典鍵
pyDict = py.dict(pyargs('a', 1, 'b', 2, 'c', 3));
keys = cell(pyDict.keys());
for i = 1:length(keys)
    key = keys{i};
    value = pyDict{key};
    fprintf('%s: %d\n', char(key), int64(value));
end
```

### 錯誤處理

```matlab
try
    result = py.some_module.function_that_might_fail();
catch ME
    if isa(ME, 'matlab.exception.PyException')
        disp('發生 Python 錯誤：');
        disp(ME.message);
    else
        rethrow(ME);
    end
end
```

## 從 Python 呼叫 MATLAB

### 設定 MATLAB 引擎

```python
# 安裝 MATLAB Engine API for Python
# 在 MATLAB 中：cd(fullfile(matlabroot,'extern','engines','python'))
# 然後：python setup.py install

import matlab.engine

# 啟動 MATLAB 引擎
eng = matlab.engine.start_matlab()

# 或連接到共享會話（MATLAB：matlab.engine.shareEngine）
eng = matlab.engine.connect_matlab()

# 列出可用會話
matlab.engine.find_matlab()
```

### 呼叫 MATLAB 函數

```python
import matlab.engine

eng = matlab.engine.start_matlab()

# 呼叫內建函數
result = eng.sqrt(16.0)
result = eng.sin(3.14159 / 2)

# 多輸出
mean_val, std_val = eng.std([1, 2, 3, 4, 5], nargout=2)

# 矩陣運算
A = matlab.double([[1, 2], [3, 4]])
B = eng.inv(A)
C = eng.mtimes(A, B)  # 矩陣乘法

# 呼叫自訂函數（必須在 MATLAB 路徑上）
result = eng.myfunction(arg1, arg2)

# 清理
eng.quit()
```

### 資料轉換（Python 到 MATLAB）

```python
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

# Python 到 MATLAB 類型
matlab_double = matlab.double([1.0, 2.0, 3.0])
matlab_int = matlab.int32([1, 2, 3])
matlab_complex = matlab.double([1+2j, 3+4j], is_complex=True)

# 2D 陣列
matlab_matrix = matlab.double([[1, 2, 3], [4, 5, 6]])

# numpy 到 MATLAB
np_array = np.array([[1, 2], [3, 4]], dtype=np.float64)
matlab_array = matlab.double(np_array.tolist())

# 使用 numpy 資料呼叫 MATLAB
result = eng.sum(matlab.double(np_array.flatten().tolist()))
```

### 非同步呼叫

```python
import matlab.engine

eng = matlab.engine.start_matlab()

# 非同步呼叫
future = eng.sqrt(16.0, background=True)

# 做其他工作...

# 準備好時取得結果
result = future.result()

# 檢查是否完成
if future.done():
    result = future.result()

# 如需要則取消
future.cancel()
```

## 常見工作流程

### 在 MATLAB 中使用 Python 函式庫

```matlab
% 從 MATLAB 使用 scikit-learn
sklearn = py.importlib.import_module('sklearn.linear_model');

% 準備資料
X = rand(100, 5);
y = X * [1; 2; 3; 4; 5] + randn(100, 1) * 0.1;

% 轉換為 Python/numpy
X_py = py.numpy.array(X);
y_py = py.numpy.array(y);

% 訓練模型
model = sklearn.LinearRegression();
model.fit(X_py, y_py);

% 取得係數
coefs = double(model.coef_);
intercept = double(model.intercept_);

% 預測
y_pred = double(model.predict(X_py));
```

### 在 Python 腳本中使用 MATLAB

```python
import matlab.engine
import numpy as np

# 啟動 MATLAB
eng = matlab.engine.start_matlab()

# 使用 MATLAB 的最佳化
def matlab_fmincon(objective, x0, A, b, Aeq, beq, lb, ub):
    """MATLAB fmincon 的包裝器。"""
    # 轉換為 MATLAB 類型
    x0_m = matlab.double(x0.tolist())
    A_m = matlab.double(A.tolist()) if A is not None else matlab.double([])
    b_m = matlab.double(b.tolist()) if b is not None else matlab.double([])

    # 呼叫 MATLAB（假設 objective 是 MATLAB 函數）
    x, fval = eng.fmincon(objective, x0_m, A_m, b_m, nargout=2)

    return np.array(x).flatten(), fval

# 使用 MATLAB 的繪圖
def matlab_plot(x, y, title_str):
    """使用 MATLAB 建立繪圖。"""
    eng.figure(nargout=0)
    eng.plot(matlab.double(x.tolist()), matlab.double(y.tolist()), nargout=0)
    eng.title(title_str, nargout=0)
    eng.saveas(eng.gcf(), 'plot.png', nargout=0)

eng.quit()
```

### 在 MATLAB 和 Python 之間共享資料

```matlab
% 為 Python 儲存資料
data = rand(100, 10);
labels = randi([0 1], 100, 1);
save('data_for_python.mat', 'data', 'labels');

% 在 Python 中：
% import scipy.io
% mat = scipy.io.loadmat('data_for_python.mat')
% data = mat['data']
% labels = mat['labels']

% 載入來自 Python 的資料（用 scipy.io.savemat 儲存）
loaded = load('data_from_python.mat');
data = loaded.data;
labels = loaded.labels;

% 替代方案：使用 CSV 進行簡單資料交換
writematrix(data, 'data.csv');
% Python：pd.read_csv('data.csv')

% Python 寫入：df.to_csv('results.csv')
results = readmatrix('results.csv');
```

### 使用 MATLAB 中不可用的 Python 套件

```matlab
% 範例：使用 Python 的 requests 函式庫
requests = py.importlib.import_module('requests');

% 發送 HTTP 請求
response = requests.get('https://api.example.com/data');
status = int64(response.status_code);

if status == 200
    data = response.json();
    % 轉換為 MATLAB 結構體
    dataStruct = struct(data);
end

% 範例：使用 Python 的 PIL/Pillow 進行進階影像處理
PIL = py.importlib.import_module('PIL.Image');

% 開啟影像
img = PIL.open('image.png');

% 調整大小
img_resized = img.resize(py.tuple({int64(256), int64(256)}));

% 儲存
img_resized.save('image_resized.png');
```
