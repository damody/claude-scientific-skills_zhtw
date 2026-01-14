# Qiskit 設定和安裝

## 安裝

使用 uv 安裝 Qiskit：

```bash
uv pip install qiskit
```

若需視覺化功能：

```bash
uv pip install "qiskit[visualization]" matplotlib
```

## Python 環境設定

建立並啟動虛擬環境以隔離相依性：

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

## 支援的 Python 版本

請查看 [Qiskit PyPI 頁面](https://pypi.org/project/qiskit/) 了解目前支援的 Python 版本。截至 2025 年，Qiskit 通常支援 Python 3.8+。

## IBM Quantum 帳戶設定

要在真實的 IBM Quantum 硬體上執行電路，您需要 IBM Quantum 帳戶和 API 令牌。

### 建立帳戶

1. 造訪 [IBM Quantum Platform](https://quantum.ibm.com/)
2. 註冊免費帳戶
3. 前往您的帳戶設定以取得 API 令牌

### 配置認證

儲存您的 IBM Quantum 憑證：

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# 儲存憑證（僅首次需要）
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN"
)

# 後續會話 - 載入已儲存的憑證
service = QiskitRuntimeService()
```

### 環境變數方法

或者，將 API 令牌設為環境變數：

```bash
export QISKIT_IBM_TOKEN="YOUR_IBM_QUANTUM_TOKEN"
```

## 本地開發（無需帳戶）

您可以使用模擬器在本地建構和測試量子電路，無需 IBM Quantum 帳戶：

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 使用模擬器在本地執行
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
```

## 驗證安裝

測試您的安裝：

```python
import qiskit
print(qiskit.__version__)

from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
print("Qiskit 安裝成功！")
```
