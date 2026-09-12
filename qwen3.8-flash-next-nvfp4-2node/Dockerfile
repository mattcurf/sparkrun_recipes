FROM vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

COPY patch_ple_mixed_precision.py /tmp/patch_ple_mixed_precision.py
COPY patch_modelopt_mtp.py /tmp/patch_modelopt_mtp.py

RUN python3 /tmp/patch_ple_mixed_precision.py \
    && python3 /tmp/patch_modelopt_mtp.py \
    && python3 -m py_compile \
        /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py \
    && rm /tmp/patch_ple_mixed_precision.py /tmp/patch_modelopt_mtp.py
