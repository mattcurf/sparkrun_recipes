FROM vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

COPY patch_ple_mixed_precision.py /tmp/patch_ple_mixed_precision.py
RUN python3 /tmp/patch_ple_mixed_precision.py && rm /tmp/patch_ple_mixed_precision.py
