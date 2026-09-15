# 两分支审查（2026-09-15）

远端克隆后核对：

- main: `8d4e254a5c906e5df28843aaef9b8ad212803744`
- corner-strategy: `0f1bfda124da14016fed483bd03db0a9fdb2c9fe`
- 工作分支 `feat/ppo` 从后者创建。已有 `results/`、`runs/`、原始环境、DQN 更新、旧训练入口和旧测试均不改写。

## 具体发现

1. [main DQN](https://github.com/fishee82oo/2048-rl/blob/8d4e254a5c906e5df28843aaef9b8ad212803744/agents/dqn_agent.py#L35) 的探索只从合法动作中抽取，利用时把其他 Q 值置为负无穷；[target 更新](https://github.com/fishee82oo/2048-rl/blob/8d4e254a5c906e5df28843aaef9b8ad212803744/agents/dqn_agent.py#L64) 却在全部四个动作上取最大值。无效动作没有执行反馈，但其估值可能进入 bootstrap。corner-strategy 保留该行为，且 target 不考虑保角过滤。本实现没有修正 DQN target，等步数实验明确标为 original Vanilla/Strategy DQN。
2. [策略模块](https://github.com/fishee82oo/2048-rl/blob/0f1bfda124da14016fed483bd03db0a9fdb2c9fe/utils/strategy.py#L44) 使用 `8*(Phi(next)-Phi(now))` 加 `+8/-16` 保角转移奖励，且终局不清零势能。它不等同于带 gamma 的 potential shaping。过滤器在锚定右上角时优先保留仍能锚定最大块的合法动作，全部不满足时回退全部合法动作；模拟不消耗 spawn RNG。奖励和过滤在旧 `--strategy` 下耦合，辅助指标不能证明策略网络本身学会保角。
3. [原训练入口](https://github.com/fishee82oo/2048-rl/blob/0f1bfda124da14016fed483bd03db0a9fdb2c9fe/train.py#L57) 以单局训练最高分选择 best，容易选择偶然有利的方块序列；episodes 预算也不能保证环境步数相等。新 PPO 和独立 DQN budget harness 都用独立验证种子的平均原始分数选择 best；DQN 算法、奖励、epsilon schedule、更新频率原样保留。选择协议变化单独注明，不冒充旧历史训练流程。
4. [旧评估入口](https://github.com/fishee82oo/2048-rl/blob/0f1bfda124da14016fed483bd03db0a9fdb2c9fe/evaluate.py#L16) 默认 seed=42 和 best_model，默认种子可能与训练重叠。每方法使用相同环境种子序列是合理的配对起点，但不同动作会产生不同后续棋盘；旧全局策略 RNG 是按方法重置，前一局长度也可能影响后一局的随机 tie-breaking。新入口逐局隔离环境和策略种子、写出 seed 列表和 checkpoint SHA256、拒绝与 checkpoint 中验证种子重叠的测试种子，分别列出训练种子与过滤条件。
5. 两分支 MLP 都是 log2 flatten16 输入、两层 128 ReLU；动作序不变。`corner-strategy` 环境仅新增无副作用的 `simulate_action` helper。此次没有改动 `env/game_2048.py` 的计分、合并、spawn 或终局逻辑。
6. 原始权重字典加载默认为 Vanilla DQN；含 `model_state_dict/strategy` 的旧 checkpoint 保留策略元数据。新 PPO 通过显式 `algorithm='ppo'` 调度，不用网络形状猜算法；显式 `--algorithm` 不匹配会报错。

## 测试审查

main 的 10 个已有测试单独在该 SHA 的 archive 上执行通过。corner-strategy 的 19 个已有测试覆盖原环境、DQN target 和策略过滤/奖励/旧 checkpoint；均保留并通过。新增测试覆盖 PPO mask、ratio、clip、GAE、时间截断、多环境、预算、奖励记账、更新和精确续训。测试日志见本目录。

历史 CSV 只作背景。本目录所有新数值来自新运行，不与原历史数据混合。
