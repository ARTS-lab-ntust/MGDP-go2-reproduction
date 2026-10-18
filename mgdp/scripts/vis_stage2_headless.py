# -*- coding: utf-8 -*-
"""
MGDP 第二階段預訓練模型的 headless 評估（雲端無螢幕可用）

和原本的 vis_stage2.py / play_stage2.py 的差別：
  1. headless，不開 Isaac Gym 視窗；深度圖 / 高度圖比較面板改成存 PNG，不開 OpenCV 視窗
  2. 10 種地形 × 10 個等級全部測（原本 play_stage2.py 只建 6 列，測不到第 6-9 級）
  3. 每個機器人固定在指定的 (等級, 地形)，不跑課程升降級；只跑一個回合就停
  4. 判定：從 x≈1 m 出發，走到該塊地形的 x >= 9 m 且途中沒有被重置（掉落、碰撞）= 通過
     另外記錄 MGDP 自己的升級標準（前進超過 5 m）供對照
  5. 結果存到 ~/mgdp_eval/：trials.csv（每次嘗試）、summary.json / summary.txt（每種地形的最高通過等級）

用法（在 ~/MGDP 下，已 source ~/mgdp_env.sh）：
  python legged_gym/scripts/vis_stage2_headless.py 2>&1 | tee ~/mgdp_eval.log

可用環境變數調整（不加就用預設值）：
  MGDP_EVAL_REPEATS=3      每個 (等級, 地形) 重複幾次，預設 3 → 共 300 個機器人
  MGDP_EVAL_SPEED=0.7      前進速度指令 (m/s)，與 play_stage2.py 相同
  MGDP_EVAL_SEED=1         隨機種子
  MGDP_EVAL_SAVE_VIS=1     是否存深度圖 / 高度圖比較 PNG（1 = 存，0 = 不存）
  MGDP_EVAL_OUT=~/mgdp_eval 結果資料夾
"""
import os

# Available dog names: a1, aliengo, anymal_c, b1, go1, go2, lite3, mini_cheetah, mini_point, solo, spot
DOG_NAMES = ["go2"]
os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

import isaacgym  # noqa: F401  （必須在 torch 之前 import）
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403  （註冊 random_dog_stage1 / stage2）
from legged_gym.envs.random_dog.random_dog import Randomdog
from legged_gym.utils import get_args, task_registry, update_cfg_from_args

import csv
import json
import math
import random
import time

import cv2
import numpy as np
import torch

OUT_DIR = os.path.expanduser(os.environ.get("MGDP_EVAL_OUT", "~/mgdp_eval"))
REPEATS = int(os.environ.get("MGDP_EVAL_REPEATS", "3"))
SPEED = float(os.environ.get("MGDP_EVAL_SPEED", "0.7"))
SEED = int(os.environ.get("MGDP_EVAL_SEED", "1"))
SAVE_VIS = os.environ.get("MGDP_EVAL_SAVE_VIS", "1") == "1"

NUM_ROWS, NUM_COLS = 10, 10          # 10 個等級（沿 x）× 10 種地形（沿 y）
SUCCESS_X = 9.0                      # 該塊地形的局部 x 座標 (m)；地形長 10 m，起點約 x = 1 m
MGDP_MOVE_UP_DIST = 5.0              # MGDP 課程的升級標準：前進超過 env_length / 2
PASS_RATIO = 2.0 / 3.0               # 「通過」某一級：成功率 >= 2/3

# MGDP 原始碼中的 terrain.idx -> 名稱
TERRAIN_NAMES = {3: "single-gap", 4: "step-stone", 5: "Stones-2Rows", 7: "stones-1Rows", 8: "single-bridge",
                 13: "air-Beams", 14: "air_stone", 15: "hurdle", 16: "ramp", 17: "corridor"}
RESULT_NAMES = ["running", "success", "fell_or_contact", "timeout"]


# ---------------------------------------------------------------------------
# 比較面板：不開視窗，只每隔一段時間存一張 PNG（最多各 25 張）
# ---------------------------------------------------------------------------
_vis_counter = {"depth": 0, "height": 0}


def _make_png_saver(kind):
    def _save(self, combined_bgr):
        n = _vis_counter[kind]
        _vis_counter[kind] += 1
        if n % 40 == 0 and n // 40 < 25:
            d = os.path.join(OUT_DIR, "vis_" + kind)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "%03d.png" % (n // 40)), combined_bgr)
    return _save


Randomdog.show_depth_compare_panel = _make_png_saver("depth")
Randomdog.show_height_compare_panel = _make_png_saver("height")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args = get_args()
    args.task = 'random_dog_stage2'
    args.num_envs = NUM_ROWS * NUM_COLS * REPEATS
    args.headless = True
    cuda = "cuda:0"
    args.rl_device = cuda
    args.render_device = cuda
    args.sim_device = cuda
    args.graphics_device_num = 0
    args.checkpoint_model = 'last.pt'
    args.load_world_model_policy = True
    args.update_wm = False
    args.compare_depth_vis = SAVE_VIS
    args.compare_height_vis = SAVE_VIS
    args.vis_env_id = 0
    args.algo = 'MGDP'
    args.seed = SEED
    model_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'models/MGDP/stage2/resume')
    args.output_name = model_dir
    args.resume_name = model_dir
    args.load_world_model_path = model_dir

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ---- 與 play_stage2.py 相同的評估設定，但改成 10 級全測 ----
    env_cfg.camera.update_wm = False
    env_cfg.camera.load_world_model_policy = True
    env_cfg.camera.load_world_model_policy_file = model_dir
    env_cfg.camera.render_compare_pre_vis = SAVE_VIS
    env_cfg.camera.render_compare_pre_map = SAVE_VIS
    env_cfg.terrain.num_cols_half = NUM_COLS
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = NUM_ROWS
    env_cfg.terrain.num_cols = NUM_COLS
    env_cfg.terrain.max_init_terrain_level = NUM_ROWS - 1
    env_cfg.terrain.curriculum = True        # 地形必須用 curriculum 模式產生，才會是「列 = 等級、欄 = 地形」的網格
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False
    env_cfg.commands.curriculum = False      # 速度指令固定，不讓指令課程放寬範圍
    env_cfg.commands.ranges.lin_vel_x = [SPEED, SPEED]
    env_cfg.commands.ranges.new_lin_vel_x = [SPEED, SPEED]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    # 8 m / 0.7 m/s ≈ 11.4 s；回合至少 25 s，避免還沒走完就逾時
    env_cfg.env.episode_length_s = max(float(getattr(env_cfg.env, 'episode_length_s', 20.0)), 25.0)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

    print(f"[eval] num_envs={args.num_envs}  repeats={REPEATS}  speed={SPEED} m/s  out={OUT_DIR}")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ---- 把每個機器人固定到指定的 (等級, 地形)，之後重置也不再換地形 ----
    env.cfg.terrain.curriculum = False
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    types = ids % NUM_COLS
    levels = (ids // NUM_COLS) % NUM_ROWS
    env.terrain_types[:] = types
    env.terrain_levels[:] = levels
    env.env_origins[:] = env.terrain_origins[levels, types]
    env.env_class[:] = env.terrain_class[levels, types]
    obs_dict = env.reset()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                          train_cfg=train_cfg, log_root=None)
    policy, _ = ppo_runner.get_inference_policy(device=env.device)
    obs_dict = env.reset()

    # ---- 跑一個回合 ----
    tile_len = float(env.terrain.env_length)

    def x_local():
        return env.root_states[:, 0] - env.env_origins[:, 0] + tile_len / 2

    start_x = x_local().clone()
    max_x = start_x.clone()
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    success = torch.zeros(n, dtype=torch.bool, device=env.device)
    reason = torch.zeros(n, dtype=torch.long, device=env.device)
    steps_taken = torch.zeros(n, dtype=torch.long, device=env.device)
    max_steps = int(env.max_episode_length) + 5
    print(f"[eval] dt={env.dt:.3f} s, 最多 {max_steps} 步（{max_steps * env.dt:.0f} s）")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    for step in range(max_steps):
        with torch.no_grad():
            actions = policy(obs_dict)
        obs_dict, _, dones, _ = env.step(actions.detach())
        reset_now = dones.bool()
        timeout_now = env.time_out_buf.bool()

        xl = x_local()
        alive = (~done) & (~reset_now)
        max_x = torch.where(alive, torch.maximum(max_x, xl), max_x)
        steps_taken[~done] += 1

        new_success = alive & (max_x >= SUCCESS_X)
        success |= new_success
        reason[new_success] = 1
        done |= new_success

        ended = (~done) & reset_now
        reason[ended & timeout_now] = 3
        reason[ended & ~timeout_now] = 2
        done |= ended

        if step % 100 == 0:
            print(f"[eval] step {step:5d}  完成 {int(done.sum())}/{n}  成功 {int(success.sum())}  "
                  f"{(time.time() - t0) / (step + 1) * 1000:.1f} ms/步")
        if bool(done.all()):
            break
    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # ---- 整理結果 ----
    rows = []
    for i in range(n):
        lv, ty = int(levels[i]), int(types[i])
        cls = int(env.terrain_class[lv, ty].item())
        progress = float(max_x[i] - start_x[i])
        rows.append(dict(env=i, terrain=TERRAIN_NAMES.get(cls, str(cls)), terrain_col=ty, level=lv,
                         repeat=i // (NUM_ROWS * NUM_COLS), success=bool(success[i]),
                         progress_m=round(progress, 2), reached_5m=progress >= MGDP_MOVE_UP_DIST,
                         result=RESULT_NAMES[int(reason[i])], time_s=round(float(steps_taken[i]) * env.dt, 2)))

    need = math.ceil(PASS_RATIO * REPEATS - 1e-9)
    summary = {}
    for ty in range(NUM_COLS):
        trs = [r for r in rows if r["terrain_col"] == ty]
        name = trs[0]["terrain"]
        per_level, per_level_5m = [], []
        for lv in range(NUM_ROWS):
            lr = [r for r in trs if r["level"] == lv]
            per_level.append(sum(r["success"] for r in lr))
            per_level_5m.append(sum(r["reached_5m"] for r in lr))
        passed = [lv for lv in range(NUM_ROWS) if per_level[lv] >= need]
        passed_5m = [lv for lv in range(NUM_ROWS) if per_level_5m[lv] >= need]
        consecutive = None
        for lv in range(NUM_ROWS):
            if per_level[lv] < need:
                break
            consecutive = lv
        summary[name] = dict(success_per_level=per_level, reached_5m_per_level=per_level_5m,
                             highest_passed=max(passed) if passed else None,
                             consecutive_from_0=consecutive,
                             highest_passed_5m=max(passed_5m) if passed_5m else None)

    info = dict(num_envs=n, repeats=REPEATS, speed_mps=SPEED, seed=SEED, steps=step + 1,
                sim_seconds=round((step + 1) * env.dt, 1), wall_seconds=round(elapsed, 1),
                ms_per_step=round(elapsed / (step + 1) * 1000, 2), torch_peak_mem_gb=round(peak_mem_gb, 2),
                pass_rule=f"成功 >= {need}/{REPEATS}；成功 = 走到 x >= {SUCCESS_X} m 且未被重置")

    with open(os.path.join(OUT_DIR, "trials.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(info=info, summary=summary), f, ensure_ascii=False, indent=2)

    lines = ["", "=" * 86, " MGDP stage2 預訓練模型（Go2）評估結果", "=" * 86,
             f" {info['pass_rule']}；表中為每一級的成功次數（共 {REPEATS} 次）", "",
             f" {'地形':<15}" + "".join(f"L{lv:<4}" for lv in range(NUM_ROWS)) + "最高  連續  (5m標準)"]
    for name, s in summary.items():
        cells = "".join(f"{c}/{REPEATS:<3}" for c in s["success_per_level"])
        hp = "-" if s["highest_passed"] is None else str(s["highest_passed"])
        cc = "-" if s["consecutive_from_0"] is None else str(s["consecutive_from_0"])
        h5 = "-" if s["highest_passed_5m"] is None else str(s["highest_passed_5m"])
        lines.append(f" {name:<15}{cells}{hp:<6}{cc:<6}{h5}")
    lines += ["", f" 模擬 {info['sim_seconds']} s，實際 {info['wall_seconds']} s，{info['ms_per_step']} ms/步，"
                  f"torch 顯存峰值 {info['torch_peak_mem_gb']} GB（不含 PhysX）", "=" * 86, ""]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[eval] 結果已存到 {OUT_DIR}")


if __name__ == '__main__':
    main()
