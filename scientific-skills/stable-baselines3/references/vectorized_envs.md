# Stable Baselines3 中的向量化環境

本文件提供 Stable Baselines3 中向量化環境的完整資訊，用於高效的平行訓練。

## 概述

向量化環境將多個獨立環境實例堆疊成單一環境，以批次方式處理動作和觀測。您不是一次與一個環境互動，而是同時與 `n` 個環境互動。

**優點：**
- **速度：** 平行執行顯著加速訓練
- **樣本效率：** 更快收集更多樣化的經驗
- **必需用於：** 幀堆疊和正規化包裝器
- **更適合：** 在線策略演算法（PPO、A2C）

## VecEnv 類型

### DummyVecEnv

在當前 Python 程序上順序執行環境。

```python
from stable_baselines3.common.vec_env import DummyVecEnv

# 方法 1：使用 make_vec_env
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=4, vec_env_cls=DummyVecEnv)

# 方法 2：手動建立
def make_env():
    def _init():
        return gym.make("CartPole-v1")
    return _init

env = DummyVecEnv([make_env() for _ in range(4)])
```

**何時使用：**
- 輕量級環境（CartPole、簡單網格）
- 當多處理程序開銷 > 計算時間時
- 除錯（更容易追蹤錯誤）
- 單執行緒環境

**效能：** 無實際平行性（順序執行）。

### SubprocVecEnv

在獨立程序中執行每個環境，實現真正的平行性。

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
```

**何時使用：**
- 計算密集型環境（物理模擬、3D 遊戲）
- 當環境計算時間足以抵消多處理程序開銷時
- 當需要真正的平行執行時

**重要：** 使用 forkserver 或 spawn 時需要將程式碼包裝在 `if __name__ == "__main__":` 中：

```python
if __name__ == "__main__":
    env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=100000)
```

**效能：** 跨 CPU 核心的真正平行性。

## 使用 make_vec_env 快速設置

建立向量化環境最簡單的方式：

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 基本使用
env = make_vec_env("CartPole-v1", n_envs=4)

# 使用 SubprocVecEnv
env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

# 使用自訂環境 kwargs
env = make_vec_env(
    "MyEnv-v0",
    n_envs=4,
    env_kwargs={"difficulty": "hard", "max_steps": 500}
)

# 使用自訂種子
env = make_vec_env("CartPole-v1", n_envs=4, seed=42)
```

## 與標準 Gym 的 API 差異

向量化環境的 API 與標準 Gym 環境不同：

### reset()

**標準 Gym：**
```python
obs, info = env.reset()
```

**VecEnv：**
```python
obs = env.reset()  # 僅返回觀測（numpy 陣列）
# 透過 env.reset_infos 存取 info（字典列表）
infos = env.reset_infos
```

### step()

**標準 Gym：**
```python
obs, reward, terminated, truncated, info = env.step(action)
```

**VecEnv：**
```python
obs, rewards, dones, infos = env.step(actions)
# 返回 4 元組而非 5 元組
# dones = terminated | truncated
# actions 是形狀為 (n_envs,) 或 (n_envs, action_dim) 的陣列
```

### 自動重置

**VecEnv 在回合結束時自動重置環境：**

```python
obs = env.reset()  # 形狀：(n_envs, obs_dim)
for _ in range(1000):
    actions = env.action_space.sample()  # 形狀：(n_envs,)
    obs, rewards, dones, infos = env.step(actions)
    # 如果 dones[i] 為 True，環境 i 已自動重置
    # 重置前的最終觀測可透過 infos[i]["terminal_observation"] 取得
```

### 終端觀測

當回合結束時，存取真正的最終觀測：

```python
obs, rewards, dones, infos = env.step(actions)

for i, done in enumerate(dones):
    if done:
        # obs[i] 已經是重置後的觀測
        # 真正的終端觀測在 info 中
        terminal_obs = infos[i]["terminal_observation"]
        print(f"回合結束，終端觀測：{terminal_obs}")
```

## 使用向量化環境訓練

### 在線策略演算法（PPO、A2C）

在線策略演算法從向量化中獲益顯著：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 建立向量化環境
env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

# 訓練
model = PPO("MlpPolicy", env, verbose=1, n_steps=128)
model.learn(total_timesteps=100000)

# 使用 n_envs=8 和 n_steps=128：
# - 每次資料收集收集 8*128=1024 步
# - 每 1024 步更新一次
```

**經驗法則：** 在線策略方法使用 4-16 個平行環境。

### 離線策略演算法（SAC、TD3、DQN）

離線策略演算法可以使用向量化但受益較少：

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# 使用較少環境（1-4）
env = make_vec_env("Pendulum-v1", n_envs=4)

# 設定 gradient_steps=-1 以提高效率
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=1,
    gradient_steps=-1,  # 每個環境步驟執行 1 次梯度步驟（4 個環境共 4 次）
)
model.learn(total_timesteps=50000)
```

**經驗法則：** 離線策略方法使用 1-4 個平行環境。

## 向量化環境的包裝器

### VecNormalize

使用執行統計正規化觀測和獎勵。

```python
from stable_baselines3.common.vec_env import VecNormalize

env = make_vec_env("Pendulum-v1", n_envs=4)

# 使用正規化包裝
env = VecNormalize(
    env,
    norm_obs=True,        # 正規化觀測
    norm_reward=True,     # 正規化獎勵
    clip_obs=10.0,        # 裁剪正規化後的觀測
    clip_reward=10.0,     # 裁剪正規化後的獎勵
    gamma=0.99,           # 獎勵正規化的折扣因子
)

# 訓練
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000)

# 儲存模型和正規化統計
model.save("ppo_pendulum")
env.save("vec_normalize.pkl")

# 載入用於評估
env = make_vec_env("Pendulum-v1", n_envs=1)
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False  # 評估時不更新統計
env.norm_reward = False  # 評估時不正規化獎勵

model = PPO.load("ppo_pendulum", env=env)
```

**何時使用：**
- 連續控制任務（特別是 MuJoCo）
- 當觀測尺度變化很大時
- 當獎勵變異數很高時

**重要：**
- 統計不會與模型一起儲存 - 需分開儲存
- 評估時停用訓練和獎勵正規化

### VecFrameStack

堆疊多個連續幀的觀測。

```python
from stable_baselines3.common.vec_env import VecFrameStack

env = make_vec_env("PongNoFrameskip-v4", n_envs=8)

# 堆疊 4 幀
env = VecFrameStack(env, n_stack=4)

# 現在觀測形狀為：(n_envs, n_stack, height, width)
model = PPO("CnnPolicy", env)
model.learn(total_timesteps=1000000)
```

**何時使用：**
- Atari 遊戲（堆疊 4 幀）
- 需要速度資訊的環境
- 部分可觀測性問題

### VecVideoRecorder

錄製代理行為的影片。

```python
from stable_baselines3.common.vec_env import VecVideoRecorder

env = make_vec_env("CartPole-v1", n_envs=1)

# 錄製影片
env = VecVideoRecorder(
    env,
    video_folder="./videos/",
    record_video_trigger=lambda x: x % 2000 == 0,  # 每 2000 步錄製
    video_length=200,  # 最大影片長度
    name_prefix="training"
)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

**輸出：** `./videos/` 目錄中的 MP4 影片。

### VecCheckNan

檢查觀測和獎勵中的 NaN 或無限值。

```python
from stable_baselines3.common.vec_env import VecCheckNan

env = make_vec_env("CustomEnv-v0", n_envs=4)

# 添加 NaN 檢查（用於除錯）
env = VecCheckNan(env, raise_exception=True, warn_once=True)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

**何時使用：**
- 除錯自訂環境
- 捕捉數值不穩定性
- 驗證環境實作

### VecTransposeImage

將圖像觀測從 (height, width, channels) 轉置為 (channels, height, width)。

```python
from stable_baselines3.common.vec_env import VecTransposeImage

env = make_vec_env("PongNoFrameskip-v4", n_envs=4)

# 將 HWC 轉換為 CHW 格式
env = VecTransposeImage(env)

model = PPO("CnnPolicy", env)
```

**何時使用：**
- 當環境返回 HWC 格式的圖像時
- SB3 的 CNN 策略期望 CHW 格式

## 進階使用

### 自訂 VecEnv

建立自訂向量化環境：

```python
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

class CustomVecEnv(DummyVecEnv):
    def step_wait(self):
        # 在步進前/後的自訂邏輯
        obs, rewards, dones, infos = super().step_wait()
        # 修改觀測/獎勵等
        return obs, rewards, dones, infos
```

### 環境方法呼叫

在包裝的環境上呼叫方法：

```python
env = make_vec_env("MyEnv-v0", n_envs=4)

# 在所有環境上呼叫方法
env.env_method("set_difficulty", "hard")

# 在特定環境上呼叫方法
env.env_method("reset_level", indices=[0, 2])

# 從所有環境取得屬性
levels = env.get_attr("current_level")
```

### 設定屬性

```python
# 在所有環境上設定屬性
env.set_attr("difficulty", "hard")

# 在特定環境上設定屬性
env.set_attr("max_steps", 1000, indices=[1, 3])
```

## 效能優化

### 選擇環境數量

**在線策略（PPO、A2C）：**
```python
# 一般規則：4-16 個環境
# 更多環境 = 更快的資料收集
n_envs = 8
env = make_vec_env("CartPole-v1", n_envs=n_envs)

# 調整 n_steps 以維持相同的資料收集長度
# 每次資料收集的總步數 = n_envs * n_steps
model = PPO("MlpPolicy", env, n_steps=128)  # 8*128 = 1024 步/資料收集
```

**離線策略（SAC、TD3、DQN）：**
```python
# 一般規則：1-4 個環境
# 更多環境幫助不大（回放緩衝區提供多樣性）
n_envs = 4
env = make_vec_env("Pendulum-v1", n_envs=n_envs)

model = SAC("MlpPolicy", env, gradient_steps=-1)  # 每個環境步驟 1 次梯度步驟
```

### CPU 核心利用

```python
import multiprocessing

# 使用比總核心數少一的數量（留一個給 Python 主程序）
n_cpus = multiprocessing.cpu_count() - 1
env = make_vec_env("MyEnv-v0", n_envs=n_cpus, vec_env_cls=SubprocVecEnv)
```

### 記憶體考量

```python
# 大型回放緩衝區 + 多個環境 = 高記憶體使用
# 如果記憶體受限，減少緩衝區大小
model = SAC(
    "MlpPolicy",
    env,
    buffer_size=100_000,  # 從 1M 減少
)
```

## 常見問題

### 問題："Can't pickle local object"

**原因：** SubprocVecEnv 需要可序列化的環境。

**解決方案：** 在類別/函數外定義環境建立：

```python
# 不好
def train():
    def make_env():
        return gym.make("CartPole-v1")
    env = SubprocVecEnv([make_env for _ in range(4)])

# 好
def make_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":
    env = SubprocVecEnv([make_env for _ in range(4)])
```

### 問題：單一環境和向量化環境之間行為不同

**原因：** 向量化環境中的自動重置。

**解決方案：** 正確處理終端觀測：

```python
obs, rewards, dones, infos = env.step(actions)
for i, done in enumerate(dones):
    if done:
        terminal_obs = infos[i]["terminal_observation"]
        # 如果需要處理 terminal_obs
```

### 問題：SubprocVecEnv 比 DummyVecEnv 慢

**原因：** 環境太輕量（多處理程序開銷 > 計算）。

**解決方案：** 簡單環境使用 DummyVecEnv：

```python
# 對於 CartPole，使用 DummyVecEnv
env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=DummyVecEnv)
```

### 問題：使用 SubprocVecEnv 時訓練崩潰

**原因：** 環境未正確隔離或有共享狀態。

**解決方案：**
- 確保環境沒有共享的全域狀態
- 將程式碼包裝在 `if __name__ == "__main__":` 中
- 使用 DummyVecEnv 進行除錯

## 最佳實踐

1. **使用適當的 VecEnv 類型：**
   - DummyVecEnv：簡單環境（CartPole、基本網格）
   - SubprocVecEnv：複雜環境（MuJoCo、Unity、3D 遊戲）

2. **為向量化調整超參數：**
   - 在回調中將 `eval_freq`、`save_freq` 除以 `n_envs`
   - 在線策略演算法維持相同的 `n_steps * n_envs`

3. **儲存正規化統計：**
   - 始終與模型一起儲存 VecNormalize 統計
   - 評估時停用訓練

4. **監控記憶體使用：**
   - 更多環境 = 更多記憶體
   - 如果需要，減少緩衝區大小

5. **先用 DummyVecEnv 測試：**
   - 更容易除錯
   - 在平行化前確保環境正常運作

## 範例

### 基本訓練迴圈

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 建立向量化環境
env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

# 訓練
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# 評估
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
```

### 使用正規化

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# 建立並正規化
env = make_vec_env("Pendulum-v1", n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 訓練
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000)

# 儲存兩者
model.save("model")
env.save("vec_normalize.pkl")

# 載入用於評估
eval_env = make_vec_env("Pendulum-v1", n_envs=1)
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load("model", env=eval_env)
```

## 額外資源

- 官方 SB3 VecEnv 指南：https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- VecEnv API 參考：https://stable-baselines3.readthedocs.io/en/master/common/vec_env.html
- 多處理程序最佳實踐：https://docs.python.org/3/library/multiprocessing.html
