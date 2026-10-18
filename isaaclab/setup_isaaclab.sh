#!/usr/bin/env bash
# =============================================================================
# 雲端 GPU 一鍵安裝：Isaac Sim 5.1.0 + Isaac Lab 2.3.2（Ubuntu 22.04 / 24.04）
# 用法：bash setup_isaaclab.sh
# 已內建本機安裝時遇到的所有修正（setuptools / flatdict / stable-baselines3 / torchaudio）
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$HOME/IsaacLab"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="env_isaaclab"

echo "========== [1/8] 檢查 GPU 與驅動 =========="
nvidia-smi
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER_MAJOR" -ge 570 ]; then TORCH_CUDA="cu128"; else TORCH_CUDA="cu118"; fi
echo ">> 驅動主版本 $DRIVER_MAJOR，PyTorch 使用 $TORCH_CUDA"

echo "========== [2/8] 安裝系統套件 =========="
sudo apt-get update -y
sudo apt-get install -y git cmake build-essential tmux wget libglu1-mesa

echo "========== [3/8] 安裝 Miniconda =========="
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "========== [4/8] 建立 Python 3.11 環境 =========="
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

echo "========== [5/8] 安裝 PyTorch 2.7.0 與 Isaac Sim 5.1.0（需要一段時間）=========="
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "========== [6/8] 下載並安裝 Isaac Lab 2.3.2 =========="
if [ ! -d "$ISAACLAB_DIR" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
git checkout v2.3.2
# 修正：新版 setuptools 移除了 pkg_resources，flatdict 會裝不起來
pip install "setuptools<81"
pip install flatdict==4.0.1 --no-build-isolation
./isaaclab.sh --install
# 修正：鎖定與 torch 2.7 相容的版本，並確保 torch 沒被換掉
pip install "stable-baselines3==2.7.0"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "========== [7/8] 放入評估程式、設定環境變數 =========="
mkdir -p "$ISAACLAB_DIR/scripts/course_eval"
cp "$PROJECT_DIR/course_eval.py" "$ISAACLAB_DIR/scripts/course_eval/course_eval.py"
if ! grep -q "OMNI_KIT_ACCEPT_EULA" "$HOME/.bashrc"; then
  {
    echo 'export OMNI_KIT_ACCEPT_EULA=YES'
    echo "source $CONDA_DIR/etc/profile.d/conda.sh"
    echo "conda activate $ENV_NAME"
  } >> "$HOME/.bashrc"
fi
export OMNI_KIT_ACCEPT_EULA=YES

echo "========== [8/8] 驗證 =========="
python -c "import torch; print('torch', torch.__version__, 'CUDA 可用:', torch.cuda.is_available())"
echo ">> 啟動空場景測試（最多 10 分鐘，第一次會比較久）..."
timeout 600 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless 2>&1 | tee /tmp/isaaclab_check.log || true
if grep -q "Setup complete" /tmp/isaaclab_check.log; then
  echo ""
  echo "=============================================="
  echo " 安裝成功！接下來執行：bash $PROJECT_DIR/run_robot.sh pretrained"
  echo "=============================================="
else
  echo ""
  echo "[錯誤] 驗證沒有通過，請把 /tmp/isaaclab_check.log 最後 50 行貼出來除錯："
  echo "       tail -n 50 /tmp/isaaclab_check.log"
  exit 1
fi
