FROM vllm/vllm-openai:qwen38-qsa-fp8-e4m3-dev
COPY overlay/cuda_communicator.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py
COPY overlay/hier_all_reduce.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/hier_all_reduce.py
RUN python3 -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py \
  /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/hier_all_reduce.py
