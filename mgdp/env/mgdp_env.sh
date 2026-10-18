deactivate 2>/dev/null
source ~/miniconda_gym/etc/profile.d/conda.sh
conda activate MGDP
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTORCH_JIT=0
