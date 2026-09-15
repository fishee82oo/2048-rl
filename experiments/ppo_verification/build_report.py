"""Regenerate the measured report from saved run evidence, without training."""
from pathlib import Path
import json
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
runs=pd.read_csv(ROOT/'matrix8192/per_training_seed.csv')
base=pd.read_csv(ROOT/'matrix8192/baselines/evaluation.csv')
initial=pd.read_csv(ROOT/'initial_policy/summary.csv')
main=runs[~runs.evaluation_condition_changed]
labels={'dqn_vanilla':'原版 Vanilla DQN','dqn_strategy':'原版 Strategy DQN',
        'ppo_raw_corner0':'PPO raw，无过滤','ppo_raw_corner1':'PPO raw，有过滤',
        'ppo_potential_corner0':'PPO potential，无过滤','ppo_potential_corner1':'PPO potential，有过滤'}
lines=['# PPO 实现与短程实验报告', '',
'## 结论', '',
'**实现、测试和有限对照已完成；尚不能据此断言 PPO 已优于充分训练的 DQN。** '
'本次固定 8,192 步预算下，raw PPO 无过滤的观测分数高于两种短训 DQN，'
'但这是早期训练比较，既不证明收敛后的优势，也不证明稳定达到 2048。所有方法的 2048 达成率均为 0。', '',
'## 协议与范围', '',
'- 18 个学习运行：6 个方法 × 训练种子 42/43/44，每次严格 8,192 个训练环境交互。',
'- 4,096/8,192 步用验证种子 100000–100009 评估，按平均原始分数选 best；最终测试固定为 200000–200099，各模型 100 局。',
'- 选中的 checkpoint 可能来自 4,096 步；`selected_checkpoint_steps` 与运行总 `training_steps=8192` 分开记录。训练耗时取同目录 last checkpoint 的完整运行计数。',
'- PPO 测试为 masked argmax；DQN 为 greedy Q；Random 随机合法动作，Greedy 随机打破同分。带过滤的 9 个模型再各做 100 局无过滤测试。',
'- 共 2,900 局主矩阵测试（1,800 模型默认条件 + 900 过滤关闭 + 200 Random/Greedy）。验证局不计入训练步数，所有学习方法采用相同验证协议。',
'- 额外完成 4,096 步 smoke、续训到 4,103 步、5 局随机采样重载评估、1 局播放；另测三个未训练 PPO 初始化各 100 局作为诊断。',
'- 模型结构/优化/奖励参数见保存的 config；无 CNN、无超参数搜索、无 DQN Bellman 修正。PPO scale=.01；DQN 保留原始 scale=1。',
'- Python 3.12.14 / PyTorch 2.14.0 / NumPy 2.5.3，macOS arm64，CPU 单线程；运行环境报告 CUDA/MPS 均不可用。锁文件已保存。', '',
'## 原始分数与大方块达成率', '',
'学习方法的 ± 是**三个训练种子均分之间的样本标准差**；中位数和局内 SD 列为各训练种子统计量的平均。达成率先逐训练种子计算，再等权平均。Random/Greedy 各只有一组 100 局，不伪造训练方差。', '',
'| 方法 | 平均分 ± 训练种子 SD | 中位数 | 局内分数 SD | ≥512 | ≥1024 | ≥2048 |',
'|---|---:|---:|---:|---:|---:|---:|']
for _,r in base.iterrows():
    lines.append(f'| {r.agent} | {r.average_score:.2f} | {r.median_score:.1f} | {r.score_std:.1f} | {r.reach_512_pct:.2f}% | {r.reach_1024_pct:.2f}% | {r.reach_2048_pct:.2f}% |')
for method in labels:
    f=main[main.method==method]
    lines.append(f'| {labels[method]} | {f.average_score.mean():.2f} ± {f.average_score.std():.2f} | {f.median_score.mean():.1f} | {f.score_std.mean():.1f} | {f.reach_512_pct.mean():.2f}% | {f.reach_1024_pct.mean():.2f}% | {f.reach_2048_pct.mean():.2f}% |')
lines+=['','### 不合并训练种子的结果','','| 方法 | seed 42 均分 | seed 43 均分 | seed 44 均分 |','|---|---:|---:|---:|']
for method in labels:
    f=main[main.method==method].set_index('training_seed')
    lines.append('| '+labels[method]+' | '+' | '.join(f'{f.loc[s,"average_score"]:.2f}' for s in (42,43,44))+' |')
lines+=['','完整的逐局分数、最大方块分布、512/1024/2048 比率、分位数及辅助指标在 '
'[逐训练种子汇总](matrix8192/per_training_seed.csv)、[跨训练种子汇总](matrix8192/across_training_seeds.csv) '
'和每个 `seed*/test/evaluation_games.csv`。没有将全部游戏池化来隐藏训练方差。', '',
'### 关闭保角过滤（评估条件变化）', '',
'| 原训练配置 | 保留过滤均分 | 关闭过滤均分 ± 训练种子 SD | 关闭过滤 ≥512 |',
'|---|---:|---:|---:|']
for method in ('dqn_strategy','ppo_raw_corner1','ppo_potential_corner1'):
    a=main[main.method==method]
    b=runs[(runs.method==method)&runs.evaluation_condition_changed]
    lines.append(f'| {labels[method]} | {a.average_score.mean():.2f} | {b.average_score.mean():.2f} ± {b.average_score.std():.2f} | {b.reach_512_pct.mean():.2f}% |')
lines+=['','过滤提高辅助保角指标不等于提高游戏分数。这里 potential PPO 关闭过滤后的均分更高，但三个种子的波动较大，不能据此定论过滤有害。', '',
'## Smoke、测试与诊断', '',
'- main 原有 10 个测试通过；当前分支 19 个原有测试 + 12 个 PPO 测试，共 **31 项通过**，见 [测试日志](tests.log)。',
'- 覆盖单合法动作和非法动作零概率、全零 mask 拒绝、保存 mask 的 ratio、一正一负 advantage 的 clip 分支、手算 GAE、自然终局、rollout bootstrap、时间截断最终 observation、多环境边界、参数变化/有限梯度、KL early stop、奖励记账和旧 DQN/PPO checkpoint 调度。',
'- 更新边界续训与连续训练的最终参数逐位一致；仅保证相同软件/设备/rollout 分段。尾部预算验证从 4096 到 4103，恰好增加 7 步，没有补足到 8 或 2048。',
'- Smoke raw 奖励 0.01 缩放后训练；2048/4096 步分别约 8699.8/10465.8 步/秒。更新前 ratio 最大误差 1.2e-7/2.4e-7。KL 约 .00421/.00144；梯度、loss 有限。',
'- 重载 PPO 播放：seed=300000，原始分数 2492，最大方块 256，215 步。旧 DQN 播放也完成（1620 分、128、172 步）。均只是一局流程验证。',
'- 最终源码重新训练 raw PPO seed42 到 8192 步，与矩阵原运行的模型参数和环境 board/score/RNG 逐位一致，见 [复现检查](reproduction.json)。单环境 legacy scale=1 的额外 257 步运行严格在 101/202/257 步验证。',
'- 初始化诊断（raw、无过滤、argmax）：seed 42/43/44 的未训练均分分别为 '+', '.join(f'{r.average_score:.2f}' for _,r in initial.iterrows())+'。短训 raw PPO 对应三个种子的测试均分均上升，但最终测试是事后诊断，不能作为后续调参选择集。', '',
'### 最后一次更新的价值诊断', '',
'| PPO 配置 | 三种子 explained variance | 三种子 value loss |',
'|---|---|---|']
for method in [m for m in labels if m.startswith('ppo')]:
    last=[json.loads((ROOT/f'matrix8192/{method}/seed{s}/updates.jsonl').read_text().splitlines()[-1]) for s in (42,43,44)]
    lines.append('| '+labels[method]+' | '+', '.join(f'{x["explained_variance"]:.3f}' for x in last)+' | '+', '.join(f'{x["value_loss"]:.3f}' for x in last)+' |')
lines+=['','价值预测的 explained variance 大多接近零或为负，短程 value learning 尚弱；未见 NaN 或梯度爆炸，不能把稳定运行误当成收敛。', '',
'## 实测耗时及正式实验预估', '',
'训练秒数为 trainer 循环计时，含验证，不含进程启动/网络初始化/最后一次保存。单次短跑受计时噪声影响；以下外推不是保证。', '',
'| 方法 | 8192 步训练秒数（三种子范围） | 1M 步粗估 |','|---|---:|---:|']
for method in labels:
    f=main[main.method==method]
    lo,hi=f.training_seconds.min(),f.training_seconds.max()
    lines.append(f'| {labels[method]} | {lo:.2f}–{hi:.2f}s | {lo/8192*1e6/60:.1f}–{hi/8192*1e6/60:.1f} 分钟 |')
total=main.training_seconds.sum()/8192*1e6/60
lines+=['',f'全 18 个 1M 步学习运行按短跑总耗时线性外推约 **{total:.0f} 分钟**，另加最终评估、启动和保存；保守预留约 1.5–2 小时，并先观察首个正式运行速度。正式验证频率更低，后期游戏长度、硬件和学习更新代价也可能改变速度。未启动正式 1M 预算。', '',
'直接执行的正式矩阵、单独消融、续训及评估命令在 [README](../../README.md#ppo-clip-and-reproducible-experiments)。实际已运行命令见 [COMMANDS](COMMANDS.md)。', '',
'## 下一步实验与限制', '',
'1. 先保持结构/超参数不变，完成 1M 步 × 三种子的等预算矩阵；确认价值 explained variance 是否改善，以及优势是否仍存在。不要根据这次测试种子继续调参后又报告它为独立测试。',
'2. 若 value learning 仍弱，在新的验证集合上比较 reward scale=.01 与 .001（其他条件固定），同时查看原始回报、value loss、EV、KL 和梯度；不要同时换 CNN/奖励/过滤而失去归因。',
'3. raw 与 potential、过滤开关的差异目前不能建立稳定因果结论。增加训练种子及独立复核种子，报告运行级不确定性；必要时将随机采样和 argmax 作为预先登记的两种评估条件。',
'4. 本次未训练修正版 DQN，原未掩码 target 的局限仍然存在。若研究 corrected DQN，独立命名、独立实验，不能回写本报告的历史基线。',
'5. 没有 2048 成功证据；≥1024 只在少量游戏出现，不构成可靠的大数字策略。未承诺 PPO 最终一定更好。', '',
'## 产物与可复现性', '',
'- 代码在 `feat/ppo`；两基线 SHA 与逐文件审查见 [AUDIT](AUDIT.md)。历史文件保持不变。',
'- [依赖锁](requirements-lock.txt)、[训练时源码摘要](source_manifest.json)、运行 metadata、commands、逐局 CSV 和诊断 JSONL 已保存。源码摘要记录短程矩阵运行时版本；之后仅加强 KL 停止诊断、算法 tag 验证、精确验证间隔及区分选中 checkpoint/完成预算的报告字段，并再次通过测试。',
'- 二进制 checkpoint 位于各实验的 `checkpoints/best.pt` / `last.pt`，遵循仓库原有策略不纳入 Git；本地保留，可用保存命令重建。',
'- 表格中的 score/达成率全部读取实际 CSV，未引用或拼接历史 results。', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
