# -*- coding: utf-8 -*-
"""
四足機器人測試場地評估腳本
適用：Isaac Sim 5.1 + Isaac Lab 2.3.x（Windows / Linux 皆可）

測試項目：
  1. 平地：前進、後退、左移、右移各 5 m（前後 >= 0.6 m/s、左右 >= 0.3 m/s），左右原地旋轉一圈（>= 90 deg/s）
  2. 門檻：高度 0.05 / 0.10 / 0.15 / 0.20 m，寬 1.6 m（>= 1.2 m），依序跨越
  3. 樓梯：階高 0.18 m、踏深 0.24 m、寬 1.2 m、8 階，上樓 -> 平台 -> 下樓
  4. MGDP 場地（--course mgdp）：MGDP 論文第二階段的 10 種極限地形 × 10 個難度等級，
     每塊 10 m × 4 m，地形直接沿用 MGDP 原始碼的產生方式（純 numpy 移植，已驗證與原始碼一致）。
     每塊從 x = 1 m 起步、只給前進指令，走到 x >= 9 m 且未掉落算通過，回報每種地形的最高通過等級。

可選機器人（皆為 Isaac Lab 內建）：
  go2       Unitree Go2
  anymal_d  ANYbotics ANYmal-D
  a1        Unitree A1

用法（Linux 雲端，在 ~/IsaacLab 下，已 conda activate env_isaaclab）：
  ./isaaclab.sh -p scripts/course_eval/course_eval.py --robot go2 --headless --checkpoint <model_xxx.pt>
  加上 --video 會把每一項測試錄成 mp4（存在 logs/course_eval/ 底下）

MGDP 場地：
  ./isaaclab.sh -p scripts/course_eval/course_eval.py --robot go2 --headless --checkpoint <model_xxx.pt> --course mgdp
  只測部分地形或等級（網格較小、啟動較快）：
    --mgdp_terrains single-gap ramp corridor --mgdp_levels 0 3 6 9
"""

import argparse

from isaaclab.app import AppLauncher

ROBOT_TASKS = {
    "go2": "Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    "anymal_d": "Isaac-Velocity-Rough-Anymal-D-Play-v0",
    "a1": "Isaac-Velocity-Rough-Unitree-A1-Play-v0",
}

parser = argparse.ArgumentParser(description="四足機器人測試場地評估")
parser.add_argument("--robot", type=str, default="go2", choices=list(ROBOT_TASKS.keys()))
parser.add_argument("--checkpoint", type=str, default=None,
                    help="自行訓練的 model_xxx.pt 路徑；不填則下載 Isaac Lab 官方預訓練模型")
parser.add_argument("--tests", nargs="+", default=["flat", "threshold", "stairs"],
                    choices=["flat", "threshold", "stairs"])
parser.add_argument("--fwd_cmd", type=float, default=0.8, help="前後移動下達的速度指令 (m/s)，需 > 0.6 才有餘裕")
parser.add_argument("--lat_cmd", type=float, default=0.5, help="左右平移下達的速度指令 (m/s)，需 > 0.3 才有餘裕")
parser.add_argument("--yaw_cmd_deg", type=float, default=120.0, help="原地旋轉下達的角速度指令 (deg/s)，需 > 90")
parser.add_argument("--obstacle_speed", type=float, default=0.5, help="跨門檻/爬樓梯時的前進速度指令 (m/s)")
parser.add_argument("--video", action="store_true", default=False, help="將每一項測試錄成 mp4 影片")
parser.add_argument("--course", type=str, default="basic", choices=["basic", "mgdp"],
                    help="basic = 原本的平地/門檻/樓梯場地；mgdp = MGDP 第二階段的 10 種地形 × 10 級")
parser.add_argument("--mgdp_terrains", nargs="+",
                    default=["single-gap", "step-stone", "Stones-2Rows", "stones-1Rows", "single-bridge",
                             "air-Beams", "air_stone", "hurdle", "ramp", "corridor"],
                    choices=["single-gap", "step-stone", "Stones-2Rows", "stones-1Rows", "single-bridge",
                             "air-Beams", "air_stone", "hurdle", "ramp", "corridor"],
                    help="要測的 MGDP 地形（預設全部 10 種）")
parser.add_argument("--mgdp_levels", nargs="+", type=int, default=list(range(10)),
                    choices=list(range(10)), help="要測的難度等級 0-9（預設全部）")
parser.add_argument("--mgdp_speed", type=float, default=0.7, help="MGDP 場地的前進速度指令 (m/s)，與 MGDP 評估腳本相同")
parser.add_argument("--mgdp_timeout", type=float, default=20.0, help="每一塊地形的時間上限 (s)")
parser.add_argument("--mgdp_seed", type=int, default=1, help="地形隨機元素的種子，固定後每次場地都一樣")
parser.add_argument("--mgdp_no_roughness", action="store_true", default=False,
                    help="不加 MGDP 原本的 ±1-4 cm 地面粗糙度")
parser.add_argument("--mgdp_lateral_hold", action="store_true", default=False,
                    help="加上橫向置中修正（MGDP 原本只給前進與朝向指令，預設不加）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True  # 錄影需要離屏渲染

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下 import 必須在模擬器啟動之後
# ---------------------------------------------------------------------------
import json
import math
import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import trimesh
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  (註冊所有任務)
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# ---------------------------------------------------------------------------
# 場地尺寸（單位：m，座標為場地局部座標，x = 前進方向）
# ---------------------------------------------------------------------------
COURSE_SIZE = (30.0, 24.0)

# 平地測試區：y ∈ [0, 12]
FLAT_START = (15.0, 6.0)

# 門檻測試區
TH_Y = 15.0
TH_HEIGHTS = [0.05, 0.10, 0.15, 0.20]
TH_X = [7.0, 11.0, 15.0, 19.0]      # 每道門檻中心位置，間隔 4 m 讓機器人恢復步態
TH_DEPTH = 0.20                      # 門檻沿行進方向的厚度
TH_WIDTH = 1.6                       # 門檻寬度 (>= 1.2 m)
TH_START_X = 3.0
TH_GOAL_X = 22.0

# 樓梯測試區（上 8 階 -> 平台 -> 下 8 階）
ST_Y = 20.5
ST_RISE, ST_TREAD, ST_WIDTH, ST_N = 0.18, 0.24, 1.2, 8
ST_X0 = 5.0                          # 第一階起點
ST_PLATFORM = 1.5                    # 頂部平台長度
ST_X_END = ST_X0 + 2 * ST_TREAD * (ST_N - 1) + ST_PLATFORM
ST_TOP_H = ST_RISE * ST_N            # 1.44 m
ST_START_X = 2.5
ST_GOAL_X = ST_X_END + 1.5


def _box(center, extents):
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def course_terrain(difficulty: float, cfg: SubTerrainBaseCfg):
    """產生整個測試場地的網格。回傳 (meshes, origin)，origin 取場地局部座標 (0,0,0)。"""
    sx, sy = cfg.size
    meshes = [_box((sx / 2, sy / 2, -0.5), (sx, sy, 1.0))]  # 地面，頂面在 z = 0

    for h, x in zip(TH_HEIGHTS, TH_X):
        meshes.append(_box((x, TH_Y, h / 2), (TH_DEPTH, TH_WIDTH, h)))

    for i in range(ST_N):
        h = (i + 1) * ST_RISE
        x0 = ST_X0 + i * ST_TREAD
        x1 = ST_X_END - i * ST_TREAD
        meshes.append(_box(((x0 + x1) / 2, ST_Y, h / 2), (x1 - x0, ST_WIDTH, h)))

    return meshes, np.array([0.0, 0.0, 0.0])


@configclass
class CourseTerrainCfg(SubTerrainBaseCfg):
    function = course_terrain


# ---------------------------------------------------------------------------
# MGDP 第二階段測試地形（gap_parkour）——從 MGDP 原始碼搬過來的純 numpy 版本
# 來源：legged_gym/legged_gym/utils/new_terrains/add_extreme_gap_terrain.py
#       legged_gym/legged_gym/utils/terrain.py（Terrain.make_terrain / add_roughness）
# 只保留第二階段實際用到的 10 種地形；計算方式、參數與原始碼相同（含原始碼的怪癖），
# 不依賴 isaacgym，可直接在 Isaac Lab 環境中執行。
# ---------------------------------------------------------------------------
MGDP_HS = 0.05              # horizontal_scale [m]
MGDP_VS = 0.005             # vertical_scale [m]
MGDP_TILE_X = 10.0          # terrain_length（沿前進方向）
MGDP_TILE_Y = 4.0           # terrain_width
MGDP_NUM_ROWS = 10          # 難度等級數（difficulty = level / 10）
MGDP_DEPTH = 0.6            # 坑深
MGDP_ROUGH = (0.01, 0.04)   # add_roughness 的 height 範圍
MGDP_ROUGH_DS = 0.5         # add_roughness 的 downsampled_scale
MGDP_SLOPE_THRESHOLD = 0.75  # 假設值：MGDP base config 的 slope_treshold 未確認，取 legged_gym 預設

# 名稱 -> MGDP 原始碼中的 terrain.idx（第二階段 10 種，依 MGDP 欄位順序）
MGDP_TERRAINS = {
    "single-gap": 3, "step-stone": 4, "Stones-2Rows": 5, "stones-1Rows": 7, "single-bridge": 8,
    "air-Beams": 13, "air_stone": 14, "hurdle": 15, "ramp": 16, "corridor": 17,
}


class _SubTerrain:
    """等同 isaacgym.terrain_utils.SubTerrain（注意：width = x 方向像素數，length = y 方向像素數）。"""

    def __init__(self, width, length, vertical_scale, horizontal_scale):
        self.width = width
        self.length = length
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)


def _random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None):
    """isaacgym.terrain_utils.random_uniform_terrain 的移植。"""
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)
    heights_range = np.arange(min_height, max_height + step, step)
    hs = terrain.horizontal_scale
    ds = np.random.choice(heights_range, (int(terrain.width * hs / downsampled_scale),
                                          int(terrain.length * hs / downsampled_scale)))
    x = np.linspace(0, terrain.width * hs, ds.shape[0])
    y = np.linspace(0, terrain.length * hs, ds.shape[1])
    xu = np.linspace(0, terrain.width * hs, terrain.width)
    yu = np.linspace(0, terrain.length * hs, terrain.length)
    # 雙線性內插（等同原始碼 interp2d(kind='linear')），只用 numpy，不需要 scipy
    tmp = np.stack([np.interp(xu, x, ds[:, j].astype(float)) for j in range(ds.shape[1])], axis=1)
    z = np.rint(np.stack([np.interp(yu, y, tmp[i, :]) for i in range(tmp.shape[0])], axis=0))
    terrain.height_field_raw += z.astype(np.int16)


def _add_roughness(terrain, difficulty=1):
    max_height = (MGDP_ROUGH[1] - MGDP_ROUGH[0]) * difficulty + MGDP_ROUGH[0]
    height = random.uniform(MGDP_ROUGH[0], max_height)
    _random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005,
                            downsampled_scale=MGDP_ROUGH_DS)


# ---- 以下函式與 add_extreme_gap_terrain.py 相同 -------------------------------------
def _parkour_step_gap_terrain(terrain, difficulty, depth, platform_size=2.):
    gap_size = int(difficulty / terrain.horizontal_scale)
    gap_size = np.clip(int(gap_size), 2, 12)
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    start_y = 0
    end_y = int(terrain.length - platform_size / 8)
    start_x = platform_size
    center_x = terrain.width // 2
    terrain.height_field_raw[start_x: center_x, start_y: end_y] = -depth
    terrain.height_field_raw[start_x + gap_size: center_x - gap_size, start_y + gap_size: end_y - gap_size] = 0
    start_x = center_x + int(platform_size / 2)
    end_x = int(terrain.width)
    terrain.height_field_raw[start_x: end_x, start_y: end_y] = -depth
    terrain.height_field_raw[start_x + gap_size: end_x - gap_size, start_y + gap_size: end_y - gap_size] = 0


def _stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance_x = int(stone_distance / terrain.horizontal_scale)
    stone_distance_y = min(int(stone_distance / terrain.horizontal_scale), 1)
    max_height = min(int(max_height / terrain.vertical_scale), 40)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(1, max_height, step=4)
    start_x = 0
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    if terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            stop_y = max(0, start_y - stone_distance_y)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            while start_y < terrain.length - 10:
                stop_y = min(terrain.length - 10, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance_y
            start_x += stone_size + stone_distance_x
    x1, x2 = 0, platform_size
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0


def _stepping_two_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    stone_size = int(stone_size / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    max_height = np.clip(max_height, 0, 30)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(1, max_height, step=3)
    stone_size = np.clip(stone_size, 6, 16)
    stone_distance_gap = int(stone_distance / terrain.horizontal_scale)
    stone_distance_gap = np.clip(stone_distance_gap, 1, 6)
    stone_distance_range = np.arange(0, stone_distance_gap, step=1) + 1
    platform_y = (terrain.length - platform_size) // 2
    row1_y = int(platform_y + platform_size / 2 - stone_size - 0.1)
    row2_y = int(platform_y + platform_size / 2 + 0.1)
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size] = np.random.choice(height_range)
        terrain.height_field_raw[start_x: stop_x, row2_y: row2_y + stone_size] = np.random.choice(height_range)
        stone_distance_gap = np.random.choice(stone_distance_range)
        start_x += stone_size + stone_distance_gap
    terrain.height_field_raw[0:platform_size, platform_y:platform_y + platform_size] = 0


def _stepping_one_stones_terrain(terrain, difficulty, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    max_height = np.clip(max_height, 0, 20)
    platform_size = int(platform_size / terrain.horizontal_scale)
    if difficulty < 0.3:
        height_range = np.arange(1, max_height, step=3)
    elif 0.3 <= difficulty < 0.7:
        height_range = np.arange(1, max_height, step=4)
    else:
        height_range = np.arange(1, max_height, step=12)
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 4, 16)
    start_x = 0
    while start_x < terrain.width:
        stone_size_x = np.clip(stone_size, 12, 30)
        stone_distance_range = np.arange(12, 30, step=1)
        stone_size_y = np.random.choice(stone_distance_range)
        row1_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
        stop_x = min(terrain.width, start_x + stone_size_x)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size_y] = np.random.choice(height_range)
        start_x += stone_size + stone_distance_gap
    terrain.height_field_raw[0:platform_size, platform_y:platform_y + platform_size] = 0


def _stepping_one_bridge_terrain(terrain, stone_size, difficulty, platform_size=1., depth=1):
    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    row1_y = int(platform_y + platform_size / 2 - stone_size / 2)
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    height_range = np.arange(0, 10, step=2)
    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size] = np.random.choice(height_range)
        start_x += stone_size
    terrain.height_field_raw[0:platform_size, platform_y:platform_y + platform_size] = 0


def _stepping_air_breams_terrain(terrain, platform_size=1., depth=1):
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_size, platform_y:platform_y + platform_size] = 0


def _air_stone(terrain, difficulty, depth=2.0):
    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = int(-depth / terrain.vertical_scale)
    terrain.height_field_raw[:, start_y:end_y] = 0


def _parkour_hurdle_terrain(terrain, difficulty, stone_size, hurdle_height_range, platform_size=1.):
    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-0.6 / terrain.vertical_scale)
    row0_y = int(platform_y + platform_size / 2 - 17 / 2)
    terrain.height_field_raw[:, row0_y:row0_y + 18] = 0
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    step_height = np.random.randint(hurdle_height_min, hurdle_height_max)
    start_x = platform_size + 20
    while start_x < terrain.width - 20:
        stone_size_x = np.clip(stone_size, 12, 30)
        stone_distance_range = np.arange(17, 30, step=1)
        stop_x = min(terrain.width, start_x + stone_size_x)
        stone_size_y = np.random.choice(stone_distance_range)
        row1_y = int(platform_y + platform_size / 2 - stone_size_y / 2)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size_y] = step_height
        stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 60, 80)
        start_x += stone_size + stone_distance_gap


def _half_sloped_terrain(terrain, level_index, platform_size=2., final_platform_length=5, depth=2.0):
    platform_size = int(platform_size / terrain.horizontal_scale)
    slope_strength = 2.0 + 2.5 * level_index
    base_height = int(-depth / terrain.vertical_scale)
    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0
    up_start, up_end = platform_size, 2 * platform_size
    mid_start, mid_end = 2 * platform_size, 3 * platform_size
    down_start = 3 * platform_size
    final_start = 4 * platform_size
    xs = np.arange(up_start, up_end)
    max_height = slope_strength * (up_end - up_start)
    terrain.height_field_raw[up_start:up_end, start_y:end_y] = (slope_strength * (xs - up_start)).astype(np.int16)[:, None]
    terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = max_height
    down_end = min(down_start + (up_end - up_start), final_start)
    xs = np.arange(down_start, down_end)
    terrain.height_field_raw[down_start:down_end, start_y:end_y] = (max_height - slope_strength * (xs - down_start)).astype(np.int16)[:, None]


def _narrow_corridor(terrain, difficulty, depth=2.0):
    platform_size = int(2 / terrain.horizontal_scale)
    base_height = int(-depth / terrain.vertical_scale)
    wall_abs_height = np.random.randint(200, 251)
    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0
    center_y = terrain.length // 2
    min_gap, max_gap = 2, 8
    narrow_gap = int(max_gap - (max_gap - min_gap) * difficulty)
    narrow_gap = np.clip(narrow_gap, min_gap, max_gap)
    terrain.height_field_raw[platform_size + 20:, center_y + narrow_gap:end_y] = wall_abs_height
    terrain.height_field_raw[platform_size + 20:, start_y:center_y - narrow_gap] = wall_abs_height


def _air_beam_boxes(level):
    """get_muti_beam_trimeshes 中單一等級的部分。回傳 [(center_xyz, size_xyz)]，座標為該塊地形的局部座標。"""
    hs = MGDP_HS
    platform_size = int(2 / hs)
    tl = MGDP_TILE_X / hs
    tw = MGDP_TILE_Y / hs
    difficulty = level / (MGDP_NUM_ROWS - 1)          # 原始碼用 i/(num_rows-1)，與高度圖的 i/num_rows 不同
    stone_size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
    stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
    bream_length = int(stone_size / hs)
    stone_distance = int(stone_distance / hs)
    size_x = bream_length / 20
    step_height = 0.04 if difficulty < 0.2 else 0.05 + 0.18 * difficulty
    max_height = int(step_height / MGDP_VS) / 100
    height_range = np.arange(0, max_height, step=0.04)
    start_y = (tw - platform_size) // 2
    start_x = platform_size
    boxes = []
    while start_x < tl:
        size_y = 2 * random.randint(7, 15) / 20
        height = np.random.choice(height_range)
        center = np.round(np.array([size_x / 2 + start_x / 20, start_y / 10, height]), 3)
        boxes.append((center, np.array([size_x, size_y, 0.1])))
        start_x += bream_length + stone_distance
    return boxes


def _air_stone_box(level):
    """get_muti_stone_trimeshes 中單一等級的部分（懸空頂板）。"""
    tl = MGDP_TILE_X / MGDP_HS
    tw = MGDP_TILE_Y / MGDP_HS
    difficulty = level / (MGDP_NUM_ROWS - 1)
    height = 0.55 - (0.55 - 0.18) * difficulty
    size = np.array([4.5, np.random.randint(18, 25) / 10, np.random.randint(1, 5) / 10])
    center_x = (tl - size[0]) // 20 - 5.0             # 原始碼含 border 5 m 的偏移，這裡扣掉
    center_y = (tw - size[1] + tw) // 20 - 5.0
    pos_x = np.random.randint(1, 2) / 10
    center = np.round(np.array([center_x + pos_x, center_y, size[2] + height]), 3)
    return [(center, size)]


def mgdp_tile(name, level, roughness=True):
    """產生一塊 MGDP 地形（10 m × 4 m）。回傳 (height_field_raw[int16], boxes)。"""
    idx = MGDP_TERRAINS[name]
    difficulty = level / MGDP_NUM_ROWS
    step_height = 0.05 + 0.18 * difficulty
    depth = MGDP_DEPTH
    t = _SubTerrain(int(MGDP_TILE_X / MGDP_HS), int(MGDP_TILE_Y / MGDP_HS), MGDP_VS, MGDP_HS)
    rough = _add_roughness if roughness else (lambda *a, **k: None)
    boxes = []
    if idx == 3:
        _parkour_step_gap_terrain(t, difficulty, depth=depth, platform_size=2)
        rough(t)
    elif idx == 4:
        stones_size = 0.7 if difficulty < 0.2 else -0.5 * difficulty * difficulty + 0.7
        stone_distance = 0.05 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        _stepping_stones_terrain(t, stone_size=stones_size, stone_distance=stone_distance,
                                 max_height=step_height, platform_size=2.0, depth=depth)
        rough(t)
    elif idx == 5:
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        _stepping_two_stones_terrain(t, stone_size=stones_size, stone_distance=stone_distance,
                                     max_height=step_height, platform_size=2.0, depth=depth)
        rough(t)
    elif idx == 7:
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        _stepping_one_stones_terrain(t, difficulty=difficulty, stone_size=stones_size, stone_distance=stone_distance,
                                     max_height=step_height, platform_size=2.0, depth=depth)
        rough(t)
    elif idx == 8:
        stones_size = 0.6 if difficulty < 0.2 else -0.6 * difficulty + 0.8
        _stepping_one_bridge_terrain(t, stone_size=stones_size, difficulty=difficulty, platform_size=2.0, depth=depth)
        rough(t)
    elif idx == 13:
        _stepping_air_breams_terrain(t, platform_size=2.0, depth=depth)
        rough(t)
        boxes = _air_beam_boxes(level)
    elif idx == 14:
        _air_stone(t, difficulty, depth=depth)          # 原始碼此地形不加粗糙度
        boxes = _air_stone_box(level)
    elif idx == 15:
        hurdle_min = step_height if difficulty < 0.1 else 0.1 + 0.40 * difficulty
        hurdle_max = 0.1 + step_height if difficulty < 0.1 else 0.2 + 0.50 * difficulty
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        _parkour_hurdle_terrain(t, difficulty=difficulty, stone_size=stones_size,
                                hurdle_height_range=[hurdle_min, hurdle_max], platform_size=2.0)
        rough(t)
    elif idx == 16:
        _half_sloped_terrain(t, level_index=difficulty * 2.5, depth=depth)
        rough(t)
    elif idx == 17:
        _narrow_corridor(t, difficulty, depth=depth)    # 原始碼此地形不加粗糙度
    return t.height_field_raw, boxes


def heightfield_to_mesh(hf, horizontal_scale, vertical_scale, slope_threshold=None):
    """與 MGDP / isaacgym 的 convert_heightfield_to_trimesh 相同（陡於門檻的面改成垂直面）。"""
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    if slope_threshold is not None:
        st = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_c = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:, :] - hf[:-1, :] > st)
        move_x[1:, :] -= (hf[:-1, :] - hf[1:, :] > st)
        move_y[:, :num_cols - 1] += (hf[:, 1:] - hf[:, :-1] > st)
        move_y[:, 1:] -= (hf[:, :-1] - hf[:, 1:] > st)
        move_c[:-1, :-1] += (hf[1:, 1:] - hf[:-1, :-1] > st)
        move_c[1:, 1:] -= (hf[:-1, :-1] - hf[1:, 1:] > st)
        xx = xx + (move_x + move_c * (move_x == 0)) * horizontal_scale
        yy = yy + (move_y + move_c * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    ind0 = (np.arange(num_cols - 1)[None, :] + num_cols * np.arange(num_rows - 1)[:, None]).flatten()
    ind1, ind2 = ind0 + 1, ind0 + num_cols
    ind3 = ind2 + 1
    triangles = np.empty((2 * ind0.size, 3), dtype=np.int64)
    triangles[0::2] = np.stack([ind0, ind3, ind1], axis=1)
    triangles[1::2] = np.stack([ind0, ind2, ind3], axis=1)
    return vertices, triangles


def mgdp_selection():
    """依命令列參數決定要放哪些地形（欄，沿 y）與哪些等級（列，沿 x）。順序與 MGDP 相同。"""
    order = list(MGDP_TERRAINS.keys())
    terrains = [t for t in order if t in args_cli.mgdp_terrains]
    levels = sorted(set(args_cli.mgdp_levels))
    return terrains, levels


def mgdp_course_terrain(difficulty: float, cfg: SubTerrainBaseCfg):
    """把選到的 MGDP 地形排成網格：第 r 列（x 方向）= 第 r 個等級，第 c 欄（y 方向）= 第 c 種地形。
    懸空橫梁與頂板是額外的方塊，合併進同一個網格（/World/ground），高度掃描器才看得到。"""
    terrains, levels = mgdp_selection()
    random.seed(args_cli.mgdp_seed)
    np.random.seed(args_cli.mgdp_seed)
    nx, ny = int(MGDP_TILE_X / MGDP_HS), int(MGDP_TILE_Y / MGDP_HS)
    hf = np.zeros((len(levels) * nx, len(terrains) * ny), dtype=np.int16)
    meshes = []
    for r, lv in enumerate(levels):
        for c, name in enumerate(terrains):
            tile, boxes = mgdp_tile(name, lv, roughness=not args_cli.mgdp_no_roughness)
            hf[r * nx:(r + 1) * nx, c * ny:(c + 1) * ny] = tile
            for center, size in boxes:
                meshes.append(_box(center + np.array([r * MGDP_TILE_X, c * MGDP_TILE_Y, 0.0]), size))
    vertices, triangles = heightfield_to_mesh(hf, MGDP_HS, MGDP_VS, MGDP_SLOPE_THRESHOLD)
    meshes.insert(0, trimesh.Trimesh(vertices=vertices, faces=triangles, process=False))
    print(f"[資訊] MGDP 場地：{len(levels)} 級 × {len(terrains)} 種地形，"
          f"{len(levels) * MGDP_TILE_X:.0f} m × {len(terrains) * MGDP_TILE_Y:.0f} m，{triangles.shape[0]:,} 個三角形")
    return meshes, np.array([0.0, 0.0, 0.0])


@configclass
class MgdpTerrainCfg(SubTerrainBaseCfg):
    function = mgdp_course_terrain


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# 評估器
# ---------------------------------------------------------------------------
class CourseRunner:
    def __init__(self, env, policy, video_dir=None):
        self.env = env
        self.uenv = env.unwrapped
        self.policy = policy
        self.video_dir = video_dir
        self.writer = None
        self.frame_i = 0
        self.robot = self.uenv.scene["robot"]
        self.cmd_term = self.uenv.command_manager.get_term("base_velocity")
        self.dt = self.uenv.step_dt
        self.device = self.uenv.device
        self.origin = self.uenv.scene.terrain.terrain_origins[0, 0].clone()
        self.default_z = float(self.robot.data.default_root_state[0, 2])
        obs = env.get_observations()
        self.obs = obs[0] if isinstance(obs, tuple) else obs

    # ---- 狀態讀取 ----
    def pos(self) -> np.ndarray:
        return (self.robot.data.root_pos_w[0] - self.origin).cpu().numpy()

    def yaw(self) -> float:
        w, x, y, z = self.robot.data.root_quat_w[0].tolist()
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # ---- 控制 ----
    def step(self, vx: float, vy: float, wz: float) -> bool:
        """下達速度指令並前進一個控制步，回傳是否跌倒（觸發 termination）。"""
        self.cmd_term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=self.device)
        actions = self.policy(self.obs)
        obs, _, dones, _ = self.env.step(actions)
        self.obs = obs[0] if isinstance(obs, tuple) else obs
        if self.writer is not None:
            self.frame_i += 1
            if self.frame_i % 2 == 0:  # 每 2 個控制步存 1 張 (約 25 fps)
                frame = self.uenv.render()
                if frame is not None:
                    self.writer.append_data(np.asarray(frame)[..., :3])
        return bool(dones[0])

    # ---- 錄影 ----
    def begin_clip(self, clip_id: str):
        self.end_clip()
        if self.video_dir is None:
            return
        try:
            import imageio.v2 as imageio
            path = os.path.join(self.video_dir, f"{clip_id}.mp4")
            self.writer = imageio.get_writer(path, fps=int(round(1.0 / (2 * self.dt))), macro_block_size=1)
        except Exception as e:  # 缺少 imageio-ffmpeg 等
            print(f"[警告] 無法錄影（{e}），本項測試不錄影。")
            self.writer = None

    def end_clip(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def teleport(self, x: float, y: float, yaw: float = 0.0):
        root = self.robot.data.default_root_state[:1].clone()
        root[:, 0:3] = self.origin + torch.tensor([x, y, self.default_z], device=self.device)
        zero = torch.zeros(1, device=self.device)
        root[:, 3:7] = quat_from_euler_xyz(zero, zero, torch.full((1,), yaw, device=self.device))
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device))
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[:1], self.robot.data.default_joint_vel[:1])

    def settle(self, seconds: float = 1.5) -> bool:
        for _ in range(int(seconds / self.dt)):
            if self.step(0.0, 0.0, 0.0):
                return True
        return False

    def heading_hold(self, target_yaw: float, k: float = 1.5) -> float:
        return float(np.clip(k * wrap_to_pi(target_yaw - self.yaw()), -1.0, 1.0))

    def lateral_hold(self, target_y: float, k: float = 1.0) -> float:
        return float(np.clip(-k * (self.pos()[1] - target_y), -0.3, 0.3))

    # ---- 平地直線 ----
    def run_linear(self, name, vx, vy, dist, min_speed, timeout=25.0):
        print(f"[測試] {name} ...")
        self.teleport(*FLAT_START)
        if self.settle():
            return dict(name=name, passed=False, note="站立階段即跌倒")
        p0 = self.pos()
        direction = np.array([np.sign(vx), np.sign(vy)], dtype=float)
        t, speeds, reached_t, fell = 0.0, [], None, False
        while t < timeout:
            fell = self.step(vx, vy, self.heading_hold(0.0))
            t += self.dt
            if fell:
                break
            disp = float(np.dot(self.pos()[:2] - p0[:2], direction))
            if t > 1.0:  # 扣除起步加速
                speeds.append(float(np.dot(self.robot.data.root_lin_vel_b[0, :2].cpu().numpy(), direction)))
            if disp >= dist:
                reached_t = t
                break
        p1 = self.pos()
        perp = np.array([-direction[1], direction[0]])
        steady = float(np.mean(speeds)) if speeds else 0.0
        passed = (reached_t is not None) and (steady >= min_speed) and not fell
        return dict(
            name=name, passed=passed, fell=fell,
            distance_m=round(float(np.dot(p1[:2] - p0[:2], direction)), 3),
            time_s=round(reached_t if reached_t else t, 2),
            avg_speed_mps=round(dist / reached_t, 3) if reached_t else None,
            steady_speed_mps=round(steady, 3),
            required_mps=min_speed,
            drift_m=round(float(abs(np.dot(p1[:2] - p0[:2], perp))), 3),
        )

    # ---- 原地旋轉 ----
    def run_turn(self, name, wz, min_rate_deg, timeout=15.0):
        print(f"[測試] {name} ...")
        self.teleport(*FLAT_START)
        if self.settle():
            return dict(name=name, passed=False, note="站立階段即跌倒")
        p0 = self.pos()
        prev, total, t, rates, done_t, fell = self.yaw(), 0.0, 0.0, [], None, False
        while t < timeout:
            fell = self.step(0.0, 0.0, wz)
            t += self.dt
            if fell:
                break
            cur = self.yaw()
            total += wrap_to_pi(cur - prev)
            prev = cur
            if t > 0.5:
                rates.append(abs(math.degrees(float(self.robot.data.root_ang_vel_b[0, 2]))))
            if abs(total) >= 2 * math.pi:
                done_t = t
                break
        steady = float(np.mean(rates)) if rates else 0.0
        passed = (done_t is not None) and (steady >= min_rate_deg) and not fell
        return dict(
            name=name, passed=passed, fell=fell,
            turned_deg=round(math.degrees(abs(total)), 1),
            time_s=round(done_t if done_t else t, 2),
            avg_rate_dps=round(360.0 / done_t, 1) if done_t else None,
            steady_rate_dps=round(steady, 1),
            required_dps=min_rate_deg,
            position_drift_m=round(float(np.linalg.norm(self.pos()[:2] - p0[:2])), 3),
        )

    # ---- 門檻 ----
    def run_thresholds(self, speed, timeout=60.0):
        print("[測試] 門檻 ...")
        self.teleport(TH_START_X, TH_Y)
        if self.settle():
            return [dict(name=f"門檻 {h:.2f} m", passed=False, note="站立階段即跌倒") for h in TH_HEIGHTS]
        crossed = [None] * len(TH_HEIGHTS)
        t, fell = 0.0, False
        while t < timeout:
            fell = self.step(speed, self.lateral_hold(TH_Y), self.heading_hold(0.0))
            t += self.dt
            if fell:
                break
            x = self.pos()[0]
            for i, tx in enumerate(TH_X):
                if crossed[i] is None and x > tx + TH_DEPTH / 2 + 0.6:  # 身體完整越過
                    crossed[i] = round(t, 2)
            if x >= TH_GOAL_X:
                break
        results = []
        for i, h in enumerate(TH_HEIGHTS):
            if crossed[i] is not None:
                note = "通過"
            elif fell and all(c is not None for c in crossed[:i]):
                note = "跌倒"
            elif fell:
                note = "未測（先前已跌倒）"
            else:
                note = "卡住 / 逾時"
            results.append(dict(name=f"門檻 {h:.2f} m", passed=crossed[i] is not None, time_s=crossed[i], note=note))
        return results

    # ---- 樓梯 ----
    def run_stairs(self, speed, timeout=90.0):
        print("[測試] 樓梯 ...")
        self.teleport(ST_START_X, ST_Y)
        if self.settle():
            return [dict(name="上樓 8 階", passed=False, note="站立階段即跌倒"),
                    dict(name="下樓 8 階", passed=False, note="站立階段即跌倒")]
        up_t, down_t, t, fell, max_h = None, None, 0.0, False, 0.0
        while t < timeout:
            fell = self.step(speed, self.lateral_hold(ST_Y), self.heading_hold(0.0))
            t += self.dt
            if fell:
                break
            p = self.pos()
            ground_h = p[2] - self.default_z
            max_h = max(max_h, ground_h)
            if up_t is None and ground_h > ST_TOP_H - 0.15:
                up_t = round(t, 2)
            if up_t is not None and p[0] >= ST_GOAL_X and ground_h < 0.15:
                down_t = round(t, 2)
                break
        return [
            dict(name="上樓 8 階", passed=up_t is not None, time_s=up_t, max_height_m=round(max_h, 3),
                 note="通過" if up_t else ("跌倒" if fell else "卡住 / 逾時")),
            dict(name="下樓 8 階", passed=down_t is not None, time_s=down_t,
                 note="通過" if down_t else ("未測（未完成上樓）" if up_t is None else ("跌倒" if fell else "卡住 / 逾時"))),
        ]


    # ---- MGDP 地形 ----
    def run_mgdp(self, name, level, row, col, speed, timeout):
        """在第 row 列、第 col 欄的那塊 MGDP 地形上，從 x = 1 m 往 +x 走。
        通過：走到該塊的 x >= 9 m 且沒有掉落/翻倒（MGDP 判定掉落的標準：基座低於 z = -0.1 m）。"""
        label = f"{name} 第{level}級"
        print(f"[測試] MGDP {label} ...")
        x_off, y_mid = row * MGDP_TILE_X, col * MGDP_TILE_Y + MGDP_TILE_Y / 2
        self.teleport(x_off + 1.0, y_mid)
        if self.settle(1.0):
            return dict(name=label, terrain=name, level=level, passed=False, progress_m=0.0, note="站立階段即跌倒")
        t, best_x, fell, passed = 0.0, 1.0, False, False
        while t < timeout:
            vy = self.lateral_hold(y_mid) if args_cli.mgdp_lateral_hold else 0.0
            fell = self.step(speed, vy, self.heading_hold(0.0))
            t += self.dt
            p = self.pos()
            if fell or p[2] < -0.1:
                fell = True
                break
            best_x = max(best_x, p[0] - x_off)
            if best_x >= 9.0:
                passed = True
                break
        progress = round(best_x - 1.0, 2)
        note = "通過" if passed else ("跌倒 / 掉落" if fell else "卡住 / 逾時")
        return dict(name=label, terrain=name, level=level, passed=passed, progress_m=progress,
                    reached_5m=bool(progress >= 5.0), time_s=round(t, 2), note=note)


def summarize_mgdp(results):
    """每種地形：通過的等級、最高通過等級、從第 0 級起連續通過到第幾級。"""
    summary = {}
    for name in [t for t in MGDP_TERRAINS if any(r["terrain"] == t for r in results)]:
        rows = sorted((r for r in results if r["terrain"] == name), key=lambda r: r["level"])
        passed = [r["level"] for r in rows if r["passed"]]
        consecutive = None
        for r in rows:
            if not r["passed"]:
                break
            consecutive = r["level"]
        summary[name] = dict(levels_tested=[r["level"] for r in rows], levels_passed=passed,
                             highest_passed=max(passed) if passed else None, consecutive_from_lowest=consecutive)
    return summary


def print_mgdp_summary(summary):
    print("\n[MGDP 地形摘要]")
    print(f"  {'地形':<16}{'最高通過等級':>8}{'連續通過到':>8}   通過的等級")
    for name, s in summary.items():
        hp = "-" if s["highest_passed"] is None else s["highest_passed"]
        cc = "-" if s["consecutive_from_lowest"] is None else s["consecutive_from_lowest"]
        print(f"  {name:<16}{hp:>8}{cc:>10}   {s['levels_passed']}")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def build_env_cfg(task: str):
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)

    # 換成自訂測試場地（放在 /World/ground，高度掃描器才看得到）
    env_cfg.scene.terrain.terrain_type = "generator"
    if args_cli.course == "mgdp":
        terrains, levels = mgdp_selection()
        size = (len(levels) * MGDP_TILE_X, len(terrains) * MGDP_TILE_Y)
        sub_terrains = {"mgdp": MgdpTerrainCfg(proportion=1.0)}
    else:
        size = COURSE_SIZE
        sub_terrains = {"course": CourseTerrainCfg(proportion=1.0)}
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=size, border_width=10.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        use_cache=False, curriculum=False,
        sub_terrains=sub_terrains,
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    # 速度指令由本腳本直接控制
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.debug_vis = False

    # 關閉干擾與逾時
    env_cfg.episode_length_s = 1.0e5
    for name in ("push_robot", "base_external_force_torque"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.observations.policy.enable_corruption = False

    # 錄影用攝影機：跟著機器人移動
    if args_cli.video:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (2.5, 2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (960, 540)
    return env_cfg


def print_report(report):
    print("\n" + "=" * 70)
    print(f" 機器人：{report['robot']}    任務：{report['task']}")
    print("=" * 70)
    for group, items in report["tests"].items():
        print(f"\n[{group}]")
        for r in items:
            mark = "PASS" if r["passed"] else "FAIL"
            extra = {k: v for k, v in r.items() if k not in ("name", "passed")}
            extra = {k: v for k, v in extra.items() if k not in ("terrain", "level")}
            print(f"  {mark}  {r['name']:<12} {extra}")
    print("=" * 70 + "\n")


def main():
    task = ROBOT_TASKS[args_cli.robot]
    env_cfg = build_env_cfg(task)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    try:  # Isaac Lab 2.3 對新版 rsl_rl 的相容處理
        import importlib.metadata as metadata
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:
        pass

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task.replace("-Play", ""))
    if not resume_path or not os.path.exists(resume_path):
        print(f"[錯誤] 找不到 {task} 的模型。請先訓練，並用 --checkpoint 指定 model_xxx.pt。")
        return
    print(f"[資訊] 使用模型：{resume_path}")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join("logs", "course_eval")
    os.makedirs(out_dir, exist_ok=True)
    video_dir = None
    if args_cli.video:
        video_dir = os.path.join(out_dir, f"videos_{args_cli.robot}_{stamp}")
        os.makedirs(video_dir, exist_ok=True)

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    report = dict(robot=args_cli.robot, task=task, checkpoint=resume_path, course=args_cli.course, tests={})
    with torch.inference_mode():
        cr = CourseRunner(env, policy, video_dir)

        def clip(clip_id, fn, *a):
            cr.begin_clip(clip_id)
            try:
                return fn(*a)
            finally:
                cr.end_clip()

        if args_cli.course == "mgdp":
            terrains, levels = mgdp_selection()
            results = []
            for r, lv in enumerate(levels):
                for c, name in enumerate(terrains):
                    results.append(clip(f"mgdp_{name}_L{lv}", cr.run_mgdp, name, lv, r, c,
                                        args_cli.mgdp_speed, args_cli.mgdp_timeout))
            report["tests"]["MGDP 地形"] = results
            report["mgdp_summary"] = summarize_mgdp(results)
            report["mgdp_settings"] = dict(speed=args_cli.mgdp_speed, timeout=args_cli.mgdp_timeout,
                                           seed=args_cli.mgdp_seed, roughness=not args_cli.mgdp_no_roughness,
                                           lateral_hold=args_cli.mgdp_lateral_hold)
        if args_cli.course == "basic" and "flat" in args_cli.tests:
            yaw = math.radians(args_cli.yaw_cmd_deg)
            report["tests"]["平地"] = [
                clip("flat_forward", cr.run_linear, "前進 5 m", +args_cli.fwd_cmd, 0.0, 5.0, 0.6),
                clip("flat_backward", cr.run_linear, "後退 5 m", -args_cli.fwd_cmd, 0.0, 5.0, 0.6),
                clip("flat_left", cr.run_linear, "左移 5 m", 0.0, +args_cli.lat_cmd, 5.0, 0.3),
                clip("flat_right", cr.run_linear, "右移 5 m", 0.0, -args_cli.lat_cmd, 5.0, 0.3),
                clip("turn_left", cr.run_turn, "左轉一圈", +yaw, 90.0),
                clip("turn_right", cr.run_turn, "右轉一圈", -yaw, 90.0),
            ]
        if args_cli.course == "basic" and "threshold" in args_cli.tests:
            report["tests"]["門檻"] = clip("thresholds", cr.run_thresholds, args_cli.obstacle_speed)
        if args_cli.course == "basic" and "stairs" in args_cli.tests:
            report["tests"]["樓梯"] = clip("stairs", cr.run_stairs, args_cli.obstacle_speed)

    print_report(report)
    if "mgdp_summary" in report:
        print_mgdp_summary(report["mgdp_summary"])
    out_file = os.path.join(out_dir, f"{args_cli.robot}_{args_cli.course}_{stamp}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[資訊] 結果已存到 {os.path.abspath(out_file)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
