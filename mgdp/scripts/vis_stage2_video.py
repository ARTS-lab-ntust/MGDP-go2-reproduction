# -*- coding: utf-8 -*-
"""
MGDP 第二階段預訓練模型（Go2）錄影腳本（headless，用 Isaac Gym 相機感測器離屏渲染）

做法：每個機器人放在一塊指定的（等級, 地形）上，各掛一台跟拍相機（跟著位置移動、不跟著轉），
走一個回合，把畫面存成 mp4。畫面左上角標示地形、等級，結束時標示成功或失敗。
判定方式與 vis_stage2_headless.py 相同：走到該塊 x >= 9 m 且未被重置 = 成功。

用法（在 ~/MGDP 下，已 source ~/mgdp_env.sh）：
  python legged_gym/scripts/vis_stage2_video.py 2>&1 | tee ~/mgdp_video.log

可用環境變數：
  MGDP_VIDEO_LEVELS=5,9      要錄的等級（逗號分隔），每個等級 × 10 種地形各錄一支
  MGDP_VIDEO_W=640 MGDP_VIDEO_H=360   影片解析度
  MGDP_VIDEO_OUT=~/mgdp_videos         輸出資料夾
  MGDP_VIDEO_SEED=1
"""
import os

DOG_NAMES = ["go2"]
os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

import isaacgym  # noqa: F401  （必須在 torch 之前 import）
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry, update_cfg_from_args

import math
import random
import time

import cv2
import numpy as np
import torch

OUT_DIR = os.path.expanduser(os.environ.get("MGDP_VIDEO_OUT", "~/mgdp_videos"))
LEVELS = [int(v) for v in os.environ.get("MGDP_VIDEO_LEVELS", "5,9").split(",") if v.strip()]
W = int(os.environ.get("MGDP_VIDEO_W", "640"))
H = int(os.environ.get("MGDP_VIDEO_H", "360"))
SEED = int(os.environ.get("MGDP_VIDEO_SEED", "1"))
SPEED = 0.7
NUM_ROWS, NUM_COLS = 10, 10
SUCCESS_X = 9.0
FRAME_EVERY = 2                      # 每 2 個控制步存一張（dt = 0.02 s → 25 fps）
CAM_OFFSET = (-1.2, -2.2, 1.0)       # 相機相對於機器人的位置：左後上方，看向機器人

TERRAIN_NAMES = {3: "single-gap", 4: "step-stone", 5: "Stones-2Rows", 7: "stones-1Rows", 8: "single-bridge",
                 13: "air-Beams", 14: "air_stone", 15: "hurdle", 16: "ramp", 17: "corridor"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args = get_args()
    args.task = 'random_dog_stage2'
    args.num_envs = len(LEVELS) * NUM_COLS
    args.headless = True
    cuda = "cuda:0"
    args.rl_device = cuda
    args.render_device = cuda
    args.sim_device = cuda
    args.graphics_device_num = 0          # base_task.py 用這個值當渲染裝置，headless 也能離屏渲染
    args.checkpoint_model = 'last.pt'
    args.load_world_model_policy = True
    args.update_wm = False
    args.compare_depth_vis = False
    args.compare_height_vis = False
    args.algo = 'MGDP'
    args.seed = SEED
    model_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'models/MGDP/stage2/resume')
    args.output_name = model_dir
    args.resume_name = model_dir
    args.load_world_model_path = model_dir

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.camera.update_wm = False
    env_cfg.camera.load_world_model_policy = True
    env_cfg.camera.load_world_model_policy_file = model_dir
    env_cfg.camera.render_compare_pre_vis = False
    env_cfg.camera.render_compare_pre_map = False
    env_cfg.env.graphics_device_num = 0
    env_cfg.terrain.num_cols_half = NUM_COLS
    env_cfg.terrain.border_size = 5
    env_cfg.terrain.num_rows = NUM_ROWS
    env_cfg.terrain.num_cols = NUM_COLS
    env_cfg.terrain.max_init_terrain_level = NUM_ROWS - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [SPEED, SPEED]
    env_cfg.commands.ranges.new_lin_vel_x = [SPEED, SPEED]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    env_cfg.env.episode_length_s = max(float(getattr(env_cfg.env, 'episode_length_s', 20.0)), 25.0)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

    print(f"[video] 等級 {LEVELS} × 10 種地形 = {args.num_envs} 支影片，{W}x{H}，輸出到 {OUT_DIR}")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ---- 固定每個機器人的（等級, 地形）----
    env.cfg.terrain.curriculum = False
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    types = ids % NUM_COLS
    lv_tensor = torch.tensor(LEVELS, device=env.device, dtype=torch.long)
    levels = lv_tensor[ids // NUM_COLS]
    env.terrain_types[:] = types
    env.terrain_levels[:] = levels
    env.env_origins[:] = env.terrain_origins[levels, types]
    env.env_class[:] = env.terrain_class[levels, types]

    # ---- 每個機器人掛一台跟拍相機（FOLLOW_POSITION：跟著位置、不跟著轉）----
    dx, dy, dz = CAM_OFFSET
    yaw = math.atan2(-dy, -dx)
    pitch = math.atan2(dz, math.hypot(dx, dy))
    cam_tf = gymapi.Transform()
    cam_tf.p = gymapi.Vec3(dx, dy, dz)
    cam_tf.r = gymapi.Quat.from_euler_zyx(0.0, pitch, yaw)
    props = gymapi.CameraProperties()
    props.width, props.height = W, H
    props.horizontal_fov = 75.0
    cams = []
    for i in range(n):
        cam = env.gym.create_camera_sensor(env.envs[i], props)
        if cam < 0:
            raise RuntimeError("建立相機失敗（create_camera_sensor 回傳 -1），這台機器可能無法離屏渲染")
        body = env.gym.find_actor_rigid_body_handle(env.envs[i], env.actor_handles[i], "base")
        env.gym.attach_camera_to_body(cam, env.envs[i], body, cam_tf, gymapi.FOLLOW_POSITION)
        cams.append(cam)

    obs_dict = env.reset()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                          train_cfg=train_cfg, log_root=None)
    policy, _ = ppo_runner.get_inference_policy(device=env.device)
    obs_dict = env.reset()

    names = []
    writers = []
    fps = int(round(1.0 / (FRAME_EVERY * env.dt)))
    for i in range(n):
        lv, ty = int(levels[i]), int(types[i])
        name = TERRAIN_NAMES.get(int(env.terrain_class[lv, ty].item()), str(ty))
        names.append((name, lv))
        path = os.path.join(OUT_DIR, f"mgdp_go2_{name}_L{lv}.mp4")
        writers.append(cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)))

    tile_len = float(env.terrain.env_length)

    def x_local():
        return env.root_states[:, 0] - env.env_origins[:, 0] + tile_len / 2

    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    result = ["" for _ in range(n)]
    max_steps = int(env.max_episode_length) + 5
    last_frames = [None] * n
    t0 = time.time()

    def grab(i, label_extra=""):
        img = env.gym.get_camera_image(env.sim, env.envs[i], cams[i], gymapi.IMAGE_COLOR)
        img = np.asarray(img, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        frame = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        name, lv = names[i]
        cv2.putText(frame, f"MGDP Go2 | {name} | level {lv}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if label_extra:
            color = (60, 200, 60) if label_extra == "SUCCESS" else (60, 60, 230)
            cv2.putText(frame, label_extra, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return frame

    for step in range(max_steps):
        with torch.no_grad():
            actions = policy(obs_dict)
        obs_dict, _, dones, _ = env.step(actions.detach())
        reset_now = dones.bool()
        xl = x_local()

        if step % FRAME_EVERY == 0:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            for i in range(n):
                if done[i] or reset_now[i]:
                    continue
                frame = grab(i)
                writers[i].write(frame)
                last_frames[i] = frame

        for i in range(n):
            if done[i]:
                continue
            if (not reset_now[i]) and xl[i] >= SUCCESS_X:
                result[i] = "SUCCESS"
            elif reset_now[i]:
                result[i] = "FAIL"
            if result[i]:
                done[i] = True
                # 結尾停格 1 秒，標示成功或失敗（失敗時用重置前最後一張畫面）
                if last_frames[i] is not None:
                    end = last_frames[i].copy()
                    color = (60, 200, 60) if result[i] == "SUCCESS" else (60, 60, 230)
                    cv2.putText(end, result[i], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                    for _ in range(fps):
                        writers[i].write(end)
        if step % 100 == 0:
            print(f"[video] step {step:5d}  完成 {int(done.sum())}/{n}  {time.time() - t0:.0f} s")
        if bool(done.all()):
            break

    for i in range(n):
        if not result[i]:
            result[i] = "TIMEOUT"
        writers[i].release()
    print("\n[video] 結果：")
    for i in range(n):
        name, lv = names[i]
        print(f"  {name:<15} L{lv}  {result[i]}")
    print(f"[video] 影片已存到 {OUT_DIR}（mp4v 編碼，建議用 VLC 播放）")


if __name__ == '__main__':
    main()
