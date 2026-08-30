#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/will/build/qwen38-kernel2-native-sm86
exec >"$ROOT/build.log" 2>&1
SOURCE="$ROOT/source"
BUILD="$ROOT/cmake-build"
IMAGE=sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a
mkdir -p "$BUILD"
docker run --rm \
  --gpus all \
  --network host \
  -e TORCH_CUDA_ARCH_LIST=8.6 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e MAX_JOBS=4 \
  -e NVCC_THREADS=2 \
  -v "$SOURCE:/workspace/vllm" \
  -v "$BUILD:/workspace/build" \
  -w /workspace/vllm \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'set -euo pipefail
    apt-get update -qq
    apt-get install -y -qq git >/dev/null
    ln -sfn /usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvrtc.so.13 \
      /usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvrtc.so
    uvx --from cmake==4.4.2 cmake -S /workspace/vllm -B /workspace/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
      -DVLLM_PYTHON_PATH=/usr/local/lib/python3.12/dist-packages \
      -DCUDA_nvrtc_LIBRARY=/usr/local/cuda/lib64/libnvrtc.so \
      -DCUDA_NVRTC_LIB=/usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvrtc.so.13 \
      -DFETCHCONTENT_BASE_DIR=/workspace/build/.deps
    uvx --from cmake==4.4.2 cmake --build /workspace/build \
      --target _C_stable_libtorch -j 4'
find "$BUILD" -maxdepth 2 -name '_C_stable_libtorch*.so' -type f -print -quit > "$ROOT/extension-path.txt"
test -s "$ROOT/extension-path.txt"
sha256sum "$(cat "$ROOT/extension-path.txt")" > "$ROOT/extension.sha256"
touch "$ROOT/build.ok"
