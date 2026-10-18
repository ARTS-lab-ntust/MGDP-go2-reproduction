# MGDP（Go2）重現與 Isaac Lab 測試

用 MGDP 官方程式碼與官方預訓練權重，在雲端評估 Unitree Go2 在 10 種極限地形 × 10 個難度等級的表現；並把 MGDP 的地形移植到 Isaac Lab，作為之後 sim-to-sim 的測試場地。

> **本 repo 不包含 MGDP 原始碼。** MGDP 原始碼沒有做任何修改，請依下方步驟另外下載原版。這裡只放我們自己新增的腳本、環境設定與實驗結果。
>
> **目前所有 MGDP 成績都來自官方預訓練模型，尚未自行訓練。**

---

## 和原版 MGDP 的差異

| 項目 | 說明 |
| --- | --- |
| MGDP 原始碼 | **未修改**。使用 [arclab-hku/MGDP](https://github.com/arclab-hku/MGDP) commit `f43139d`（2026-07-08） |
| 新增腳本 | 3 個，放在 `mgdp/scripts/`，使用時複製到 `MGDP/legged_gym/scripts/` |
| 環境設定 | 官方 README 沒寫的安裝細節與環境變數，放在 `mgdp/env/` |

新增的 3 個腳本：

| 檔案 | 用途 |
| --- | --- |
| `vis_stage2_headless.py` | 無螢幕評估 stage 2 官方模型：10 種地形 × 10 級 × 每格 N 次，輸出 CSV / JSON / 文字摘要與深度圖、高度圖比較 PNG |
| `vis_stage2_video.py` | 指定等級 × 10 種地形，每格錄一支跟拍 mp4 |
| `vis_train_video.py` | 從頭跑一小段 stage 1 訓練並錄下訓練場景（只用來展示平行訓練，不是訓練成果） |

`vis_stage2_headless.py` 以官方 `legged_gym/legged_gym/scripts/play_stage2.py` 的評估設定為基礎（速度指令 0.7 m/s、關閉觀察值雜訊、摩擦與質量隨機化），差別如下：

1. 官方只建 6 列地形（等級 0～5），這裡改成 10 級全測。
2. 每個機器人固定在一種（地形, 等級），不跑課程升降級，只跑一個回合。
3. 成功判定：從 x ≈ 1 m 出發，走到該塊地形 x ≥ 9 m，且途中沒有被重置。
4. 回合長度至少 25 秒，避免還沒走完就逾時。

三個腳本都是在執行時替換部分函式（例如把 OpenCV 視窗改成存 PNG、掛上錄影相機），**不修改 MGDP 任何原始檔**。

---

## 資料夾結構

```
mgdp-go2-reproduction/
├── README.md
├── mgdp/
│   ├── scripts/        新增的 3 個評估 / 錄影腳本
│   └── env/            mgdp_env.sh、pip_freeze.txt、sysinfo.txt
├── isaaclab/
│   ├── setup_isaaclab.sh   一鍵安裝 Isaac Sim 5.1 + Isaac Lab 2.3.2
│   ├── run_robot.sh        Isaac Lab 內建四足的訓練 / 評估流程
│   └── course_eval.py      測試場地（平地、門檻、樓梯、MGDP 10 種地形）
└── results/
    ├── 2026-10-06_mgdp_official_go2/        MGDP 官方模型評估結果
    └── 2026-10-06_isaaclab_go2_mgdp_course/ Isaac Lab Go2 在 MGDP 場地的結果
```

---

## 測試環境（2026-10-06）

| 項目 | 版本 |
| --- | --- |
| 雲端 | MassedCompute（透過 NVIDIA Brev），1 × NVIDIA L40S 48 GB |
| 驅動 / CUDA | 580.126.09 / 13.0 |
| MGDP 端 | Python 3.8.20、torch 1.10.0+cu113、torchvision 0.11.1+cu113、numpy 1.23.5、warp-lang 1.4.2 |
| Isaac Lab 端 | Python 3.11、torch 2.7.0、Isaac Sim 5.1.0、Isaac Lab v2.3.2 |

完整套件清單：`mgdp/env/pip_freeze.txt`；GPU 資訊：`mgdp/env/sysinfo.txt`。

---

## 一、MGDP（Isaac Gym）安裝

> 只支援 Linux。以下每行都可以直接複製貼上。

**1. 安裝 Miniconda（裝在 `~/miniconda_gym`，和 Isaac Lab 分開）**

```bash
cd ~
U=https://repo.anaconda.com/miniconda
wget $U/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p ~/miniconda_gym
source ~/miniconda_gym/etc/profile.d/conda.sh
```

**2. 建立 Python 3.8 環境**

```bash
conda create -n MGDP -c conda-forge \
  --override-channels python=3.8.20 -y
conda activate MGDP
```

**3. 安裝 PyTorch 1.10 + CUDA 11.3**

```bash
F=https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch==1.10.0+cu113 -f $F
pip install torchvision==0.11.1+cu113 -f $F
```

**4. 下載原版 MGDP，固定在測試過的版本**

```bash
git clone https://github.com/arclab-hku/MGDP.git ~/MGDP
cd ~/MGDP && git checkout f43139d
cd ~/MGDP/isaacgym/python && pip install -e .
cd ~/MGDP/legged_gym
pip install -e . --no-deps
pip install -r requirement-gpu.txt
cd ~/MGDP/warp_sensor && pip install -e .
pip install numpy==1.23.5 ninja
```

**5. 下載本 repo，放入環境變數檔與腳本**

```bash
cd ~
git clone <本 repo 的網址> ~/repo
cp ~/repo/mgdp/env/mgdp_env.sh ~/mgdp_env.sh
cp ~/repo/mgdp/scripts/*.py ~/MGDP/legged_gym/scripts/
source ~/mgdp_env.sh
```

`mgdp_env.sh` 會啟動 conda 環境並設定三個環境變數：

| 變數 | 原因 |
| --- | --- |
| `PYTORCH_JIT=0` | torch 1.10 不認得 RTX 40 系列 / L40S（sm_89），不關 JIT 會出現 `nvrtc` 架構錯誤 |
| `VK_ICD_FILENAMES` | 指定 NVIDIA 的 Vulkan 驅動，Isaac Gym 離屏渲染需要 |
| `LD_LIBRARY_PATH` | 優先使用 conda 環境內的函式庫 |

---

## 二、MGDP 評估與錄影

每次開新的終端機都要先 `source ~/mgdp_env.sh`，並在 `~/MGDP` 底下執行。

**評估（10 種地形 × 10 級）**

```bash
source ~/mgdp_env.sh
cd ~/MGDP
python legged_gym/scripts/vis_stage2_headless.py \
  2>&1 | tee ~/mgdp_eval.log
```

可用環境變數：`MGDP_EVAL_REPEATS`（每格次數，預設 3）、`MGDP_EVAL_SPEED`（預設 0.7）、`MGDP_EVAL_SEED`、`MGDP_EVAL_SAVE_VIS`、`MGDP_EVAL_OUT`（預設 `~/mgdp_eval`）。

**錄影**

```bash
python legged_gym/scripts/vis_stage2_video.py \
  2>&1 | tee ~/mgdp_video.log
python legged_gym/scripts/vis_train_video.py \
  2>&1 | tee ~/mgdp_train_video.log
```

---

## 三、Isaac Lab 安裝與測試場地

**1. 安裝（約 30–60 分鐘）**

```bash
mkdir -p ~/quadruped_project
cp ~/repo/isaaclab/* ~/quadruped_project/
cd ~/quadruped_project
sed -i 's/\r$//' *.sh
bash setup_isaaclab.sh
```

`setup_isaaclab.sh` 會安裝 Miniconda（`~/miniconda3`）、Python 3.11 環境 `env_isaaclab`、PyTorch 2.7.0、Isaac Sim 5.1.0、Isaac Lab v2.3.2，並把 `course_eval.py` 複製到 `~/IsaacLab/scripts/course_eval/`。

**2. 在 MGDP 場地測試 Isaac Lab 的模型**

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/course_eval/course_eval.py \
  --robot go2 --headless --course mgdp \
  --checkpoint <model_xxx.pt>
```

只測部分地形或等級：加上 `--mgdp_terrains single-gap ramp --mgdp_levels 0 9`。

**3. Isaac Lab 內建四足的訓練與評估**

```bash
bash ~/quadruped_project/run_robot.sh go2 quick
bash ~/quadruped_project/run_robot.sh go2 train
```

訓練參數見 `run_robot.sh` 的 `train()`；訓練方法全部是 Isaac Lab 內建（RSL-RL PPO、187 點高度掃描），只修改速度指令設定。

---

## 結果摘要

### MGDP 官方模型（Go2），每格 10 次，通過 = 成功 ≥ 7/10

| 地形（程式名稱） | 最高通過等級（0–9） |
| --- | --- |
| single-gap（間隙） | 9 |
| step-stone（網格踏石） | 8 |
| Stones-2Rows（雙排踏石） | 9 |
| stones-1Rows（單排踏石） | 9 |
| single-bridge（單木橋） | 9 |
| air-Beams（懸空橫梁） | 8 |
| air_stone | 9 |
| hurdle（跨欄） | 9 |
| ramp（坡道） | 9 |
| corridor（窄走廊） | 6 |

- 每一級的成功次數見 `results/2026-10-06_mgdp_official_go2/eval_10x/summary.txt`；每次 3 次的版本在 `eval_3x/`。
- 窄走廊第 7–9 級全部失敗：通道寬度比 Go2 機身還窄。
- 這裡的判定方式是我們自己定的；論文 Figure 6 是「能穩定通過的最高難度 ÷ 10」，評估流程不完全相同，**不能直接拿數字對比**。

`vis_depth/`：MGDP 內建的深度圖比較（左：加雜訊的輸入、中：perception model 重建、右：乾淨深度圖）。
`vis_height/`：MGDP 內建的高度圖比較（左：預測、右：真實；17 × 11 點，涵蓋身體後方 0.4 m 到前方 1.2 m）。

### Isaac Lab 內建 Go2 模型在 MGDP 場地

自行訓練的 Isaac Lab Go2（1500 iterations）在 MGDP 第 0 級與第 9 級全部失敗。這是預期結果：該模型只在一般粗糙地形訓練過，作為之後比較的基準線。細節見 `results/2026-10-06_isaaclab_go2_mgdp_course/summary.txt`。

---

## 已知問題與注意事項

- **官方模型的步態姿勢偏低**（評估影片中看起來像拖著走），但官方 GitHub 的實機影片走得正常，原因調查中。程式中維持身體高度的 reward（`base_height`）權重為 0，是可能原因之一。
- **`course_eval.py` 存 JSON 的錯誤已修正（2026-10-18）**：`reached_5m` 原本是 numpy 的 `bool_`，`json.dump` 會中斷；已改成 `bool(...)`。2026-10-06 的 JSON 因此不完整，結果改以終端機輸出保存。
- MassedCompute 機器的使用者名稱是 `shadeform`，不是 `ubuntu`。
- Windows 上編輯過的 `.sh` 上傳到 Linux 前，要先執行 `sed -i 's/\r$//' *.sh` 去掉換行符號。
- Isaac Lab / Isaac Sim 不支援沒有 RT Core 的 GPU（A100、H100）。

---

## 來源與授權

- MGDP：[arclab-hku/MGDP](https://github.com/arclab-hku/MGDP)。MGDP repo 內含的 Isaac Gym 屬 NVIDIA 專有授權，因此本 repo 不附 MGDP 原始碼與 `isaacgym/`。
- `course_eval.py` 中的 MGDP 地形產生方式移植自 MGDP 的 `legged_gym/legged_gym/utils/new_terrains/add_extreme_gap_terrain.py`（純 numpy 改寫，已驗證與原始碼一致）。
- Isaac Lab：[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)。
