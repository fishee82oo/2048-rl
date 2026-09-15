# 实际执行命令

在仓库根目录使用独立 `.venv`。环境锁定见 `requirements-lock.txt`。

```bash
# 获取并核对分支
git clone https://github.com/fishee82oo/2048-rl.git
git rev-parse origin/main origin/corner-strategy
git switch -c feat/ppo origin/corner-strategy

# 当前分支：19 个已有测试 + PPO 测试
.venv/bin/python -m unittest discover -s tests -v

# main 另在 git archive 导出的 /private/tmp/2048-main-baseline 执行相同 unittest 命令

.venv/bin/python train_ppo.py --config configs/ppo_smoke.json \
  --output-dir experiments/ppo_verification/smoke

.venv/bin/python train_ppo.py --config configs/ppo_smoke.json \
  --resume experiments/ppo_verification/smoke/checkpoints/last.pt \
  --total-steps 4103 --output-dir experiments/ppo_verification/resume_partial

.venv/bin/python play.py --checkpoint experiments/ppo_verification/resume_partial/checkpoints/last.pt \
  --algorithm ppo --seed 300000 --delay 0

.venv/bin/python evaluate.py \
  --checkpoint experiments/ppo_verification/resume_partial/checkpoints/last.pt \
  --algorithm ppo --games 5 --seed 300000 --no-deterministic --no-baselines \
  --output-dir experiments/ppo_verification/sampled_reload

.venv/bin/python scripts/run_matrix.py --output-dir experiments/ppo_verification/matrix8192 \
  --total-steps 8192 --validation-interval 4096 --train-seeds 42 43 44 \
  --test-games 100 --device cpu --execute
```

矩阵的每个实际子命令存于 `matrix8192/commands.json`，stdout/stderr 存于 `command_00.log` 等。
新入口拒绝覆盖已有实验目录；复现时请更换 output-dir，例如 `experiments/reproduction/...`。
最初安装前运行测试由于缺少 NumPy/PyTorch 失败；安装独立环境后重新执行通过，未把依赖缺失当作代码测试通过。


追加诊断和报告生成（均已执行）：

```bash
.venv/bin/python scripts/evaluate_initial_ppo.py experiments/ppo_verification/initial_policy
.venv/bin/python scripts/summarize_matrix.py experiments/ppo_verification/matrix8192
.venv/bin/python experiments/ppo_verification/build_report.py
```

报告修订时，从各模型的 `last.pt` 读取完整训练预算/耗时，补充到已存在的 CSV 元数据列；
原选择 checkpoint 的步数保存在 `selected_checkpoint_steps`。游戏分数、动作、种子、达成率没有改写。

最终代码验证还执行了：

```bash
python train_ppo.py --config configs/ppo_smoke.json --total-steps 257 \
  --validation-interval 101 --num-envs 1 --reward-mode legacy --corner-filter \
  --reward-scale 1 --output-dir experiments/ppo_verification/legacy_single
python play.py --checkpoint experiments/ppo_verification/matrix8192/dqn_strategy/seed42/checkpoints/best.pt \
  --algorithm dqn --seed 300000 --delay 0
python train_ppo.py --config configs/ppo_formal.json --total-steps 8192 --seed 42 \
  --reward-mode raw --no-corner-filter --device cpu --validation-interval 4096 \
  --output-dir experiments/ppo_verification/reproduce_raw42
python scripts/run_matrix.py --output-dir experiments/formal_1m --total-steps 1000000 \
  --train-seeds 42 43 44 --test-games 100 --device cpu
```

实际解释器均为 `.venv/bin/python`。最后一条未带 `--execute`，只生成正式命令，没有启动正式训练。
复现运行与原 matrix seed42 的 model 张量逐位比较，环境 board/score/RNG 比较均一致，见 `reproduction.json`。
legacy 单环境恰好在 101、202、257 步验证；旧 DQN 播放得到 1620 分、最大块 128、172 步。
