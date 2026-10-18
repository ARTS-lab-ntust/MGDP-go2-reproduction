#!/usr/bin/env bash
# =============================================================================
# 四足機器人：一次只跑一隻（省錢版）
#
#   bash run_robot.sh pretrained          用官方預訓練模型評估三隻（不訓練，最便宜，約 20–30 分鐘）
#   bash run_robot.sh go2 quick           快速測試流程（10 次迭代，約 10–15 分鐘）
#   bash run_robot.sh go2 train           正式訓練 + 評估 Go2
#   bash run_robot.sh go2 eval            用「自己最近一次訓練」的 Go2 模型重新評估
#   bash run_robot.sh go2 resume 1000     從 logs/rsl_rl/<實驗名>/restored/ 的模型接續訓練 1000 次
#
# 機器人名稱：go2 / anymal_d / a1
# 可選參數：NUM_ENVS=2048 bash run_robot.sh go2 train   （顯示記憶體不足時調小）
# =============================================================================
set -uo pipefail

usage() {
  cat <<'EOF'

用法：
  bash run_robot.sh pretrained        用官方模型評估三隻（不訓練）
  bash run_robot.sh <機器人> quick     快速測試整個流程
  bash run_robot.sh <機器人> train     正式訓練 + 評估
  bash run_robot.sh <機器人> eval      用自己最近訓練好的模型重新評估
  bash run_robot.sh <機器人> resume N  接續訓練 N 次（模型放在 logs/rsl_rl/<實驗名>/restored/）

機器人：go2 / anymal_d / a1
建議順序：pretrained → go2 quick → go2 train → anymal_d train → a1 train

EOF
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
cd "$HOME/IsaacLab"

declare -A TASK=(
  [go2]="Isaac-Velocity-Rough-Unitree-Go2-v0"
  [anymal_d]="Isaac-Velocity-Rough-Anymal-D-v0"
  [a1]="Isaac-Velocity-Rough-Unitree-A1-v0"
)
declare -A EXP=(
  [go2]="unitree_go2_rough"
  [anymal_d]="anymal_d_rough"
  [a1]="unitree_a1_rough"
)

NUM_ENVS="${NUM_ENVS:-4096}"
STAMP="$(date +%Y%m%d_%H%M%S)"

new_result_dir() {  # $1 = 名稱
  RESULT_DIR="$HOME/results/${1}_$STAMP"
  mkdir -p "$RESULT_DIR"
  echo ">> 結果會存到 $RESULT_DIR"
}

evaluate() {  # $1 = 機器人, $2 = 模型路徑（空字串 = 官方模型）
  local R="$1" CKPT="$2" ARGS=()
  [ -n "$CKPT" ] && ARGS+=(--checkpoint "$CKPT")
  echo "################ [$R] 測試場地評估 ################"
  ./isaaclab.sh -p scripts/course_eval/course_eval.py \
      --robot "$R" --headless --video "${ARGS[@]}" \
      2>&1 | tee "$RESULT_DIR/eval_$R.log"
  # 把這次產生的評估結果 (json + 影片) 複製到結果資料夾
  mkdir -p "$RESULT_DIR/course_eval"
  find logs/course_eval -maxdepth 1 -newer "$RESULT_DIR" -name "${R}_*.json" -exec cp {} "$RESULT_DIR/course_eval/" \; 2>/dev/null
  find logs/course_eval -maxdepth 1 -newer "$RESULT_DIR" -type d -name "videos_${R}_*" -exec cp -r {} "$RESULT_DIR/course_eval/" \; 2>/dev/null
}

train() {  # $1 = 機器人, $2 = 最大迭代次數（空字串 = 預設 1500），$3 之後 = 額外參數（例如接續訓練）
  local R="$1" ITER="$2" ARGS=()
  shift 2
  [ -n "$ITER" ] && ARGS+=(--max_iterations "$ITER")
  ARGS+=("$@")
  echo "################ [$R] 開始訓練（環境數 $NUM_ENVS）################"
  # 2026-09-29 修正：舊版關閉 heading 模式，機器人會走圓弧、地形等級升不上去。
  # 現在保留 heading 模式給 60% 的機器人（走直線、追朝向），
  # 其餘 40% 直接給旋轉指令，範圍放寬到 ±2.0 rad/s（約 115 deg/s）以達到 90 deg/s 的要求。
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "${TASK[$R]}" --headless --num_envs "$NUM_ENVS" "${ARGS[@]}" \
      env.commands.base_velocity.heading_command=true \
      env.commands.base_velocity.rel_heading_envs=0.6 \
      "env.commands.base_velocity.ranges.ang_vel_z=[-2.0,2.0]" \
      2>&1 | tee "$RESULT_DIR/train_$R.log"
}

latest_ckpt() {  # $1 = 機器人；輸出最近一次訓練的最後一個模型
  local RUN_DIR
  RUN_DIR="$(ls -td logs/rsl_rl/${EXP[$1]}/*/ 2>/dev/null | head -1)"
  [ -z "$RUN_DIR" ] && return 1
  ls "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1
}

package() {
  tar czf "$RESULT_DIR.tar.gz" -C "$HOME/results" "$(basename "$RESULT_DIR")"
  echo ""
  echo "=============================================================="
  echo " 完成！請下載：$RESULT_DIR.tar.gz"
  echo " 下載完記得到雲端平台網頁停止或刪除機器（MassedCompute 不能暫停，只能刪除）"
  echo "=============================================================="
}

# ----------------------------------------------------------------------------
MODE="${1:-}"

if [ "$MODE" = "pretrained" ]; then
  new_result_dir "pretrained"
  for R in go2 anymal_d a1; do
    evaluate "$R" ""
  done
  package
  exit 0
fi

if [ -z "$MODE" ] || [ -z "${TASK[$MODE]+x}" ]; then
  usage; exit 1
fi

R="$MODE"
ACTION="${2:-}"

case "$ACTION" in
  quick)
    NUM_ENVS=256
    new_result_dir "${R}_quick"
    train "$R" 10
    CKPT="$(latest_ckpt "$R")" || { echo "[錯誤] 找不到模型，請看 $RESULT_DIR/train_$R.log"; exit 1; }
    evaluate "$R" "$CKPT"
    echo ">> 快速測試只訓練 10 次，結果全部 FAIL 是正常的；重點是流程要跑完、沒有紅字錯誤。"
    package
    ;;
  train)
    new_result_dir "${R}_train"
    train "$R" ""
    CKPT="$(latest_ckpt "$R")" || { echo "[錯誤] 找不到模型，請看 $RESULT_DIR/train_$R.log"; exit 1; }
    echo ">> 使用模型 $CKPT"
    mkdir -p "$RESULT_DIR/model"
    cp "$CKPT" "$RESULT_DIR/model/"
    cp -r "$(dirname "$CKPT")/params" "$RESULT_DIR/model/" 2>/dev/null || true
    evaluate "$R" "$CKPT"
    package
    ;;
  resume)
    ITER="${3:-1000}"
    RESTORE_DIR="logs/rsl_rl/${EXP[$R]}/restored"
    LAST="$(ls "$RESTORE_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)"
    if [ -z "$LAST" ]; then
      echo "[錯誤] 找不到要接續的模型，請先把 model_xxx.pt 放到 ~/IsaacLab/$RESTORE_DIR/"
      exit 1
    fi
    new_result_dir "${R}_resume"
    echo ">> 從 $LAST 接續訓練 $ITER 次"
    train "$R" "$ITER" --resume --load_run restored --checkpoint "$(basename "$LAST")"
    CKPT="$(latest_ckpt "$R")" || { echo "[錯誤] 找不到模型，請看 $RESULT_DIR/train_$R.log"; exit 1; }
    echo ">> 使用模型 $CKPT"
    mkdir -p "$RESULT_DIR/model"
    cp "$CKPT" "$RESULT_DIR/model/"
    cp -r "$(dirname "$CKPT")/params" "$RESULT_DIR/model/" 2>/dev/null || true
    evaluate "$R" "$CKPT"
    package
    ;;
  eval)
    new_result_dir "${R}_eval"
    CKPT="$(latest_ckpt "$R")" || { echo "[錯誤] 還沒有 $R 的訓練模型，請先執行：bash run_robot.sh $R train"; exit 1; }
    echo ">> 使用模型 $CKPT"
    evaluate "$R" "$CKPT"
    package
    ;;
  *)
    usage; exit 1
    ;;
esac
