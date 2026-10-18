# -*- coding: utf-8 -*-
"""
MGDP 訓練場景錄影（headless）

從頭跑一小段 stage1 訓練（MGDP 原本的 train 流程，不做任何修改），同時用兩台固定相機錄下訓練場地：
  overview：高處俯瞰整片訓練地形與上千隻 Go2
  close   ：較近的斜視角，看得清楚個別機器人
錄到指定步數就自動停止訓練（保證不會跑完 60001 個 iteration）。
注意：這是剛開始訓練的畫面，機器人還不會走，用來呈現「MGDP 怎麼平行訓練」，不是訓練成果。

用法（在 ~/MGDP 下，已 source ~/mgdp_env.sh）：
  python legged_gym/scripts/vis_train_video.py 2>&1 | tee ~/mgdp_train_video.log

可用環境變數：
  MGDP_TRAINVID_ENVS=2048     同時訓練的機器人數
  MGDP_TRAINVID_STEPS=1500    錄幾個控制步後停止（1500 步 = 模擬 30 s = 影片 30 s）
  MGDP_TRAINVID_OUT=~/mgdp_train_videos
"""
import os

DOG_NAMES = ["go2"]
os.environ["DOG_NAMES"] = ",".join(DOG_NAMES)

import isaacgym  # noqa: F401  （必須在 torch 之前 import）
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.train import *  # noqa: F401,F403  （與原本 train.py 相同的入口，提供 train / get_args）
from legged_gym.envs.random_dog.random_dog import Randomdog

import atexit
import time

import cv2
import numpy as np

OUT_DIR = os.path.expanduser(os.environ.get("MGDP_TRAINVID_OUT", "~/mgdp_train_videos"))
NUM_ENVS = int(os.environ.get("MGDP_TRAINVID_ENVS", "2048"))
MAX_STEPS = int(os.environ.get("MGDP_TRAINVID_STEPS", "1500"))
W, H = 1280, 720
FRAME_EVERY = 2                       # 每 2 個控制步存一張（dt = 0.02 s → 25 fps，約等於實際速度）

# stage1 的 mix 地形：每塊 8 m × 8 m，列（沿 x）= 難度、欄（沿 y）= 地形種類；
# 開始時機器人分布在前幾列（x 約 0-48 m）。座標為世界座標（legged_gym 的 env 間距為 0）。
CAMS = {
    "overview": ((-8.0, -8.0, 18.0), (24.0, 28.0, 0.0)),
    "close": ((4.0, 4.0, 4.0), (14.0, 14.0, 0.0)),
}

_WRITERS = []


class _StopTraining(Exception):
    pass


def _release_all():
    while _WRITERS:
        _WRITERS.pop().release()


atexit.register(_release_all)

_orig_init = Randomdog.__init__
_orig_step = Randomdog.step


def _init_with_cameras(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    os.makedirs(OUT_DIR, exist_ok=True)
    props = gymapi.CameraProperties()
    props.width, props.height = W, H
    props.horizontal_fov = 70.0
    fps = int(round(1.0 / (FRAME_EVERY * self.dt)))
    self._rec = []
    for name, (pos, tgt) in CAMS.items():
        cam = self.gym.create_camera_sensor(self.envs[0], props)
        if cam < 0:
            raise RuntimeError("建立相機失敗（create_camera_sensor 回傳 -1），這台機器可能無法離屏渲染")
        self.gym.set_camera_location(cam, self.envs[0], gymapi.Vec3(*pos), gymapi.Vec3(*tgt))
        path = os.path.join(OUT_DIR, f"mgdp_train_{name}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        _WRITERS.append(writer)
        self._rec.append((name, cam, writer))
    self._rec_step = 0
    self._rec_t0 = time.time()
    print(f"[train-video] 相機已建立，{self.num_envs} 隻 Go2，最多錄 {MAX_STEPS} 步，輸出到 {OUT_DIR}")


def _step_with_recording(self, actions):
    out = _orig_step(self, actions)
    if not getattr(self, "_rec", None):
        return out
    self._rec_step += 1
    if self._rec_step % FRAME_EVERY == 0:
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        for name, cam, writer in self._rec:
            img = self.gym.get_camera_image(self.sim, self.envs[0], cam, gymapi.IMAGE_COLOR)
            img = np.asarray(img, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
            frame = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"MGDP stage1 training | {self.num_envs} x Go2 | t = {self._rec_step * self.dt:5.1f} s",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    if self._rec_step % 250 == 0:
        print(f"[train-video] 已錄 {self._rec_step}/{MAX_STEPS} 步（{time.time() - self._rec_t0:.0f} s）")
    if self._rec_step >= MAX_STEPS:
        raise _StopTraining()
    return out


Randomdog.__init__ = _init_with_cameras
Randomdog.step = _step_with_recording


def main():
    args = get_args()
    args.task = 'random_dog_stage1'
    args.num_envs = NUM_ENVS
    args.headless = True
    cuda = "cuda:0"
    args.rl_device = cuda
    args.render_device = cuda
    args.sim_device = cuda
    args.graphics_device_num = 0
    args.seed = 1
    args.algo = 'MGDP'
    args.load_world_model_policy = False
    args.max_iterations = 200                  # 實際由 MAX_STEPS 提早停止
    args.output_name = os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs/random_dog/MGDP/stage1/train_video_demo')
    try:
        train(args)
    except _StopTraining:
        print("[train-video] 已達錄影步數，停止訓練")
    finally:
        _release_all()
    for name in CAMS:
        p = os.path.join(OUT_DIR, f"mgdp_train_{name}.mp4")
        if os.path.exists(p):
            print(f"[train-video] {p}  {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
