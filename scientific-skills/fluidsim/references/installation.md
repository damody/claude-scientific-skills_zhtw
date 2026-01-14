# FluidSim 安裝

## 需求

- Python >= 3.9
- 建議使用虛擬環境

## 安裝方法

### 基本安裝

使用 uv 安裝 fluidsim：

```bash
uv pip install fluidsim
```

### 含 FFT 支援（擬譜求解器必需）

大多數 fluidsim 求解器使用基於傅立葉的方法，需要 FFT 函式庫：

```bash
uv pip install "fluidsim[fft]"
```

這會安裝 fluidfft 和 pyfftw 依賴項。

### 含 MPI 和 FFT（用於平行模擬）

用於高效能平行計算：

```bash
uv pip install "fluidsim[fft,mpi]"
```

注意：這會觸發 mpi4py 的本地編譯。

## 環境設定

### 輸出目錄

設定環境變數以控制模擬資料的儲存位置：

```bash
export FLUIDSIM_PATH=/path/to/simulation/outputs
export FLUIDDYN_PATH_SCRATCH=/path/to/working/directory
```

### FFT 方法選擇

指定 FFT 實作（可選）：

```bash
export FLUIDSIM_TYPE_FFT2D=fft2d.with_fftw
export FLUIDSIM_TYPE_FFT3D=fft3d.with_fftw
```

## 驗證

測試安裝：

```bash
pytest --pyargs fluidsim
```

## 不需要認證

FluidSim 不需要 API 金鑰或認證令牌。
