# 為 Stable Baselines3 建立自訂環境

本指南提供建立與 Stable Baselines3 相容的自訂 Gymnasium 環境的完整資訊。

## 環境結構

### 必要方法

每個自訂環境必須繼承自 `gymnasium.Env` 並實作：

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        """初始化環境，定義 action_space 和 observation_space"""
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """將環境重置為初始狀態"""
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        info = {}
        return observation, info

    def step(self, action):
        """執行一個時間步"""
        observation = self.observation_space.sample()
        reward = 0.0
        terminated = False  # 回合自然結束
        truncated = False   # 回合因時間限制結束
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        """視覺化環境（可選）"""
        pass

    def close(self):
        """清理資源（可選）"""
        pass
```

### 方法詳細說明

#### `__init__(self, ...)`

**目的：** 初始化環境並定義空間。

**要求：**
- 必須呼叫 `super().__init__()`
- 必須定義 `self.action_space`
- 必須定義 `self.observation_space`

**範例：**
```python
def __init__(self, grid_size=10, max_steps=100):
    super().__init__()
    self.grid_size = grid_size
    self.max_steps = max_steps
    self.current_step = 0

    # 定義空間
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(
        low=0, high=grid_size-1, shape=(2,), dtype=np.float32
    )
```

#### `reset(self, seed=None, options=None)`

**目的：** 將環境重置為初始狀態。

**要求：**
- 必須呼叫 `super().reset(seed=seed)`
- 必須返回 `(observation, info)` 元組
- 觀測必須符合 `observation_space`
- Info 必須是字典（可以為空）

**範例：**
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # 初始化狀態
    self.agent_pos = self.np_random.integers(0, self.grid_size, size=2)
    self.goal_pos = self.np_random.integers(0, self.grid_size, size=2)
    self.current_step = 0

    observation = self._get_observation()
    info = {"episode": "started"}

    return observation, info
```

#### `step(self, action)`

**目的：** 在環境中執行一個時間步。

**要求：**
- 必須返回 5 元組：`(observation, reward, terminated, truncated, info)`
- 動作必須根據 `action_space` 有效
- 觀測必須符合 `observation_space`
- 獎勵應該是浮點數
- Terminated：如果回合自然結束（達到目標、失敗等）則為 True
- Truncated：如果回合因時間限制結束則為 True
- Info 必須是字典

**範例：**
```python
def step(self, action):
    # 應用動作
    self.agent_pos += self._action_to_direction(action)
    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)
    self.current_step += 1

    # 計算獎勵
    distance = np.linalg.norm(self.agent_pos - self.goal_pos)
    goal_reached = distance < 1.0

    if goal_reached:
        reward = 100.0
    else:
        reward = -distance * 0.1

    # 檢查終止條件
    terminated = goal_reached
    truncated = self.current_step >= self.max_steps

    observation = self._get_observation()
    info = {"distance": distance, "steps": self.current_step}

    return observation, reward, terminated, truncated, info
```

## 空間類型

### Discrete

用於離散動作（例如 {0, 1, 2, 3}）。

```python
self.action_space = spaces.Discrete(4)  # 4 個動作：0, 1, 2, 3
```

**重要：** SB3 不支援 `start != 0` 的 `Discrete` 空間。始終從 0 開始。

### Box（連續）

用於範圍內的連續值。

```python
# [-1, 1] 內的 1D 連續動作
self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

# 2D 位置觀測
self.observation_space = spaces.Box(
    low=0, high=10, shape=(2,), dtype=np.float32
)

# 3D RGB 圖像（通道優先格式）
self.observation_space = spaces.Box(
    low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
)
```

**圖像重要事項：**
- 必須是 `dtype=np.uint8`，範圍 [0, 255]
- 使用**通道優先**格式：(channels, height, width)
- SB3 自動透過除以 255 進行正規化
- 如果已預先正規化，在 policy_kwargs 中設定 `normalize_images=False`

### MultiDiscrete

用於多個離散變數。

```python
# 兩個離散變數：第一個有 3 個選項，第二個有 4 個選項
self.action_space = spaces.MultiDiscrete([3, 4])
```

### MultiBinary

用於二元向量。

```python
# 5 個二元標誌
self.action_space = spaces.MultiBinary(5)  # 例如 [0, 1, 1, 0, 1]
```

### Dict

用於字典觀測（例如結合圖像和感測器）。

```python
self.observation_space = spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
    "vector": spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32),
    "discrete": spaces.Discrete(3),
})
```

**重要：** 使用 Dict 觀測時，使用 `"MultiInputPolicy"` 而非 `"MlpPolicy"`。

```python
model = PPO("MultiInputPolicy", env, verbose=1)
```

### Tuple

用於元組觀測（較少見）。

```python
self.observation_space = spaces.Tuple((
    spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
    spaces.Discrete(3),
))
```

## 重要限制和最佳實踐

### 資料類型

- **觀測：** 連續值使用 `np.float32`
- **圖像：** 使用 `np.uint8`，範圍 [0, 255]
- **獎勵：** 返回 Python float 或 `np.float32`
- **Terminated/Truncated：** 返回 Python bool

### 隨機數生成

始終使用 `self.np_random` 以確保可重現性：

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    # 使用 self.np_random 而非 np.random
    random_pos = self.np_random.integers(0, 10, size=2)
    random_float = self.np_random.random()
```

### 回合終止

- **Terminated：** 自然結束（達到目標、代理死亡等）
- **Truncated：** 人為結束（時間限制、外部中斷）

```python
def step(self, action):
    # ... 環境邏輯 ...

    goal_reached = self._check_goal()
    time_limit_exceeded = self.current_step >= self.max_steps

    terminated = goal_reached  # 自然結束
    truncated = time_limit_exceeded  # 時間限制

    return observation, reward, terminated, truncated, info
```

### Info 字典

使用 info 字典進行除錯和記錄：

```python
info = {
    "episode_length": self.current_step,
    "distance_to_goal": distance,
    "success": goal_reached,
    "total_reward": self.cumulative_reward,
}
```

**特殊鍵：**
- `"terminal_observation"`：當回合結束時由 VecEnv 自動添加

## 進階功能

### Metadata

提供渲染資訊：

```python
class CustomEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # ...
```

### 渲染模式

```python
def render(self):
    if self.render_mode == "human":
        # 列印或顯示供人類觀看
        print(f"代理位於 {self.agent_pos}")

    elif self.render_mode == "rgb_array":
        # 返回 numpy 陣列 (height, width, 3) 用於錄影
        canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        # 在畫布上繪製環境
        return canvas
```

### 目標條件環境（用於 HER）

對於事後經驗回放，使用特定的觀測結構：

```python
self.observation_space = spaces.Dict({
    "observation": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
    "achieved_goal": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
    "desired_goal": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
})

def compute_reward(self, achieved_goal, desired_goal, info):
    """HER 環境所需"""
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return -distance
```

## 環境驗證

訓練前始終驗證您的環境：

```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv()
check_env(env, warn=True)
```

**常見驗證錯誤：**

1. **"Observation is not within bounds"**
   - 檢查觀測是否保持在定義的空間內
   - 確保正確的 dtype（Box 空間為 np.float32）

2. **"Reset should return tuple"**
   - 返回 `(observation, info)`，而非僅 observation

3. **"Step should return 5-tuple"**
   - 返回 `(obs, reward, terminated, truncated, info)`

4. **"Action is out of bounds"**
   - 驗證 action_space 定義是否符合預期動作

5. **"Observation/Action dtype mismatch"**
   - 確保觀測符合空間 dtype（通常為 np.float32）

## 環境註冊

向 Gymnasium 註冊您的環境：

```python
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MyCustomEnv-v0",
    entry_point="my_module:CustomEnv",
    max_episode_steps=200,
    kwargs={"grid_size": 10},  # 預設 kwargs
)

# 現在可以使用 gym.make
env = gym.make("MyCustomEnv-v0")
```

## 測試自訂環境

### 基本測試

```python
def test_environment(env, n_episodes=5):
    """使用隨機動作測試環境"""
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"回合 {episode+1}：獎勵={episode_reward:.2f}，步數={steps}")
```

### 訓練測試

```python
from stable_baselines3 import PPO

def train_test(env, timesteps=10000):
    """快速訓練測試"""
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    # 評估
    obs, info = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

## 常見模式

### 網格世界

```python
class GridWorldEnv(gym.Env):
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(0, size-1, shape=(2,), dtype=np.float32)
```

### 連續控制

```python
class ContinuousEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
```

### 基於圖像的環境

```python
class VisionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        # 通道優先：(channels, height, width)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )
```

### 多模態環境

```python
class MultiModalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "sensors": spaces.Box(-10, 10, shape=(4,), dtype=np.float32),
        })
```

## 效能考量

### 高效觀測生成

```python
# 預分配陣列
def __init__(self):
    # ...
    self._obs_buffer = np.zeros(self.observation_space.shape, dtype=np.float32)

def _get_observation(self):
    # 重複使用緩衝區而非分配新陣列
    self._obs_buffer[0] = self.agent_x
    self._obs_buffer[1] = self.agent_y
    return self._obs_buffer
```

### 向量化

使環境操作可向量化：

```python
# 好：使用 numpy 操作
def step(self, action):
    direction = np.array([[0,1], [0,-1], [1,0], [-1,0]])[action]
    self.pos = np.clip(self.pos + direction, 0, self.size-1)

# 避免：盡可能避免 Python 迴圈
# for i in range(len(self.agents)):
#     self.agents[i].update()
```

## 疑難排解

### "Observation out of bounds"
- 檢查所有觀測是否在定義的空間內
- 驗證正確的 dtype（np.float32 vs np.float64）

### "NaN or Inf in observation/reward"
- 添加檢查：`assert np.isfinite(reward)`
- 使用 `VecCheckNan` 包裝器捕捉問題

### "Policy doesn't learn"
- 檢查獎勵縮放（正規化獎勵）
- 驗證觀測正規化
- 確保獎勵訊號有意義
- 檢查探索是否充足

### "Training crashes"
- 使用 `check_env()` 驗證環境
- 檢查自訂環境中的競爭條件
- 驗證動作/觀測空間一致性

## 額外資源

- 範本：參見 `scripts/custom_env_template.py`
- Gymnasium 文件：https://gymnasium.farama.org/
- SB3 自訂環境指南：https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
