# Stable Baselines3 回調系統

本文件提供 Stable Baselines3 中回調系統的完整資訊，用於監控和控制訓練。

## 概述

回調（Callback）是在訓練期間特定時間點呼叫的函數，用於：
- 監控訓練指標
- 儲存檢查點
- 實作提早停止
- 記錄自訂指標
- 動態調整超參數
- 觸發評估

## 內建回調

### EvalCallback

定期評估代理並儲存最佳模型。

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,                                    # 獨立的評估環境
    best_model_save_path="./logs/best_model/",  # 儲存最佳模型的位置
    log_path="./logs/eval/",                    # 儲存評估日誌的位置
    eval_freq=10000,                            # 每 N 步評估一次
    n_eval_episodes=5,                          # 每次評估的回合數
    deterministic=True,                         # 使用確定性動作
    render=False,                               # 評估時是否渲染
    verbose=1,
    warn=True,
)

model.learn(total_timesteps=100000, callback=eval_callback)
```

**關鍵功能：**
- 根據平均獎勵自動儲存最佳模型
- 將評估指標記錄到 TensorBoard
- 達到獎勵閾值時可停止訓練

**重要：** 使用向量化訓練環境時，調整 `eval_freq`：
```python
# 使用 4 個平行環境時，將 eval_freq 除以 n_envs
eval_freq = 10000 // 4  # 每 10000 個總環境步驟評估一次
```

### CheckpointCallback

定期儲存模型檢查點。

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10000,                     # 每 N 步儲存
    save_path="./logs/checkpoints/",     # 檢查點目錄
    name_prefix="rl_model",              # 檢查點檔案前綴
    save_replay_buffer=True,             # 儲存回放緩衝區（僅離線策略）
    save_vecnormalize=True,              # 儲存 VecNormalize 統計
    verbose=2,
)

model.learn(total_timesteps=100000, callback=checkpoint_callback)
```

**輸出檔案：**
- `rl_model_10000_steps.zip` - 10k 步時的模型
- `rl_model_20000_steps.zip` - 20k 步時的模型
- 等等

**重要：** 向量化環境需調整 `save_freq`（除以 n_envs）。

### StopTrainingOnRewardThreshold

當平均獎勵超過閾值時停止訓練。

```python
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=200,  # 當平均獎勵 >= 200 時停止
    verbose=1,
)

# 必須與 EvalCallback 一起使用
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,  # 發現新最佳時觸發
    eval_freq=10000,
    n_eval_episodes=5,
)

model.learn(total_timesteps=1000000, callback=eval_callback)
```

### StopTrainingOnNoModelImprovement

如果模型在 N 次評估內沒有改善則停止訓練。

```python
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10,  # 10 次評估無改善後停止
    min_evals=20,                 # 停止前的最少評估次數
    verbose=1,
)

# 與 EvalCallback 一起使用
eval_callback = EvalCallback(
    eval_env,
    callback_after_eval=stop_callback,
    eval_freq=10000,
)

model.learn(total_timesteps=1000000, callback=eval_callback)
```

### StopTrainingOnMaxEpisodes

在達到最大回合數後停止訓練。

```python
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

stop_callback = StopTrainingOnMaxEpisodes(
    max_episodes=1000,  # 1000 回合後停止
    verbose=1,
)

model.learn(total_timesteps=1000000, callback=stop_callback)
```

### ProgressBarCallback

在訓練期間顯示進度條（需要 tqdm）。

```python
from stable_baselines3.common.callbacks import ProgressBarCallback

progress_callback = ProgressBarCallback()

model.learn(total_timesteps=100000, callback=progress_callback)
```

**輸出：**
```
100%|██████████| 100000/100000 [05:23<00:00, 309.31it/s]
```

## 建立自訂回調

### BaseCallback 結構

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    """
    自訂回調範本。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 自訂初始化

    def _init_callback(self) -> None:
        """
        訓練開始時呼叫一次。
        用於需要存取 model/env 的初始化。
        """
        pass

    def _on_training_start(self) -> None:
        """
        在第一次資料收集開始前呼叫。
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        在收集新樣本前呼叫（在線策略演算法）。
        """
        pass

    def _on_step(self) -> bool:
        """
        在環境中每一步後呼叫。

        返回：
            bool：如果為 False，訓練將停止。
        """
        return True  # 繼續訓練

    def _on_rollout_end(self) -> None:
        """
        在資料收集結束後呼叫（在線策略演算法）。
        """
        pass

    def _on_training_end(self) -> None:
        """
        在訓練結束時呼叫。
        """
        pass
```

### 可用屬性

在回調內部，您可以存取：

- **`self.model`**：RL 演算法實例
- **`self.training_env`**：訓練環境
- **`self.n_calls`**：`_on_step()` 被呼叫的次數
- **`self.num_timesteps`**：總環境步數
- **`self.locals`**：演算法的區域變數（因演算法而異）
- **`self.globals`**：演算法的全域變數
- **`self.logger`**：TensorBoard/CSV 記錄的日誌記錄器
- **`self.parent`**：父回調（如果在 CallbackList 中使用）

## 自訂回調範例

### 範例 1：記錄自訂指標

```python
class LogCustomMetricsCallback(BaseCallback):
    """
    將自訂指標記錄到 TensorBoard。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 檢查回合是否結束
        if self.locals["dones"][0]:
            # 記錄回合獎勵
            episode_reward = self.locals["infos"][0].get("episode", {}).get("r", 0)
            self.episode_rewards.append(episode_reward)

            # 記錄到 TensorBoard
            self.logger.record("custom/episode_reward", episode_reward)
            self.logger.record("custom/mean_reward_last_100",
                             np.mean(self.episode_rewards[-100:]))

        return True
```

### 範例 2：調整學習率

```python
class LinearScheduleCallback(BaseCallback):
    """
    在訓練期間線性降低學習率。
    """

    def __init__(self, initial_lr=3e-4, final_lr=3e-5, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr

    def _on_step(self) -> bool:
        # 計算進度（0 到 1）
        progress = self.num_timesteps / self.locals["total_timesteps"]

        # 線性內插
        new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress

        # 更新學習率
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] = new_lr

        # 記錄學習率
        self.logger.record("train/learning_rate", new_lr)

        return True
```

### 範例 3：基於移動平均的提早停止

```python
class EarlyStoppingCallback(BaseCallback):
    """
    如果獎勵的移動平均沒有改善則停止訓練。
    """

    def __init__(self, check_freq=10000, min_reward=200, window=100, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_reward = min_reward
        self.window = window
        self.rewards = []

    def _on_step(self) -> bool:
        # 收集回合獎勵
        if self.locals["dones"][0]:
            reward = self.locals["infos"][0].get("episode", {}).get("r", 0)
            self.rewards.append(reward)

        # 每 check_freq 步檢查一次
        if self.n_calls % self.check_freq == 0 and len(self.rewards) >= self.window:
            mean_reward = np.mean(self.rewards[-self.window:])
            if self.verbose > 0:
                print(f"平均獎勵：{mean_reward:.2f}")

            if mean_reward >= self.min_reward:
                if self.verbose > 0:
                    print(f"停止：達到獎勵閾值！")
                return False  # 停止訓練

        return True  # 繼續訓練
```

### 範例 4：根據自訂指標儲存最佳模型

```python
class SaveBestModelCallback(BaseCallback):
    """
    當自訂指標最佳時儲存模型。
    """

    def __init__(self, check_freq=1000, save_path="./best_model/", verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_score = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 計算自訂指標（範例：策略熵）
            custom_metric = self.locals.get("entropy_losses", [0])[-1]

            if custom_metric > self.best_score:
                self.best_score = custom_metric
                if self.verbose > 0:
                    print(f"新最佳！儲存模型到 {self.save_path}")
                self.model.save(os.path.join(self.save_path, "best_model"))

        return True
```

### 範例 5：記錄環境特定資訊

```python
class EnvironmentInfoCallback(BaseCallback):
    """
    記錄來自環境的自訂資訊。
    """

    def _on_step(self) -> bool:
        # 從環境存取 info 字典
        info = self.locals["infos"][0]

        # 記錄來自環境的自訂指標
        if "distance_to_goal" in info:
            self.logger.record("env/distance_to_goal", info["distance_to_goal"])

        if "success" in info:
            self.logger.record("env/success_rate", info["success"])

        return True
```

## 鏈接多個回調

使用 `CallbackList` 組合多個回調：

```python
from stable_baselines3.common.callbacks import CallbackList

callback_list = CallbackList([
    eval_callback,
    checkpoint_callback,
    progress_callback,
    custom_callback,
])

model.learn(total_timesteps=100000, callback=callback_list)
```

或直接傳遞列表：

```python
model.learn(
    total_timesteps=100000,
    callback=[eval_callback, checkpoint_callback, custom_callback]
)
```

## 事件驅動回調

回調可以在特定事件上觸發其他回調：

```python
from stable_baselines3.common.callbacks import EventCallback

# 達到獎勵閾值時停止訓練
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200)

# 定期評估並在發現新最佳時觸發 stop_callback
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,  # 發現新最佳模型時觸發
    eval_freq=10000,
)
```

## 記錄到 TensorBoard

使用 `self.logger.record()` 記錄指標：

```python
class TensorBoardCallback(BaseCallback):
    def _on_step(self) -> bool:
        # 記錄純量
        self.logger.record("custom/my_metric", value)

        # 記錄多個指標
        self.logger.record("custom/metric1", value1)
        self.logger.record("custom/metric2", value2)

        # 日誌記錄器自動寫入 TensorBoard
        return True
```

**在 TensorBoard 中查看：**
```bash
tensorboard --logdir ./logs/
```

## 進階模式

### 課程學習

```python
class CurriculumCallback(BaseCallback):
    """
    隨時間增加任務難度。
    """

    def __init__(self, difficulty_schedule, verbose=0):
        super().__init__(verbose)
        self.difficulty_schedule = difficulty_schedule

    def _on_step(self) -> bool:
        # 根據進度更新環境難度
        progress = self.num_timesteps / self.locals["total_timesteps"]

        for threshold, difficulty in self.difficulty_schedule:
            if progress >= threshold:
                self.training_env.env_method("set_difficulty", difficulty)

        return True
```

### 基於族群的訓練

```python
class PopulationBasedCallback(BaseCallback):
    """
    根據效能調整超參數。
    """

    def __init__(self, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.performance_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 評估效能
            perf = self._evaluate_performance()
            self.performance_history.append(perf)

            # 如果效能停滯則調整超參數
            if len(self.performance_history) >= 3:
                recent = self.performance_history[-3:]
                if max(recent) - min(recent) < 0.01:  # 檢測到停滯
                    self._adjust_hyperparameters()

        return True

    def _adjust_hyperparameters(self):
        # 範例：增加學習率
        for param_group in self.model.policy.optimizer.param_groups:
            param_group["lr"] *= 1.2
```

## 除錯提示

### 列印可用屬性

```python
class DebugCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls == 1:
            print("self.locals 中可用的內容：")
            for key in self.locals.keys():
                print(f"  {key}: {type(self.locals[key])}")
        return True
```

### 常見問題

1. **回調未被呼叫：**
   - 確保回調已傳遞給 `model.learn()`
   - 檢查 `_on_step()` 是否返回 `True`

2. **回調中的 AttributeError：**
   - 並非所有屬性在所有回調中都可用
   - 使用 `self.locals.get("key", default)` 以確保安全

3. **記憶體洩漏：**
   - 不要在回調狀態中儲存大型陣列
   - 定期清除緩衝區

4. **效能影響：**
   - 減少 `_on_step()` 中的計算（每步都會呼叫）
   - 使用 `check_freq` 限制昂貴的操作

## 最佳實踐

1. **使用適當的回調時機：**
   - `_on_step()`：用於每步都會變化的指標
   - `_on_rollout_end()`：用於在資料收集期間計算的指標
   - `_init_callback()`：用於一次性初始化

2. **高效記錄：**
   - 不要每步都記錄（會影響效能）
   - 聚合指標並定期記錄

3. **處理向量化環境：**
   - 記住 `dones`、`infos` 等是陣列
   - 檢查每個環境的 `dones[i]`

4. **獨立測試回調：**
   - 建立簡單的測試案例
   - 在長時間訓練前驗證回調行為

5. **記錄自訂回調：**
   - 清晰的文件字串
   - 在註解中提供使用範例

## 額外資源

- 官方 SB3 回調指南：https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
- 回調 API 參考：https://stable-baselines3.readthedocs.io/en/master/common/callbacks.html
- TensorBoard 文件：https://www.tensorflow.org/tensorboard
