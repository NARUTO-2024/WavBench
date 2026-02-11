FROM vllm/vllm-openai:v0.10.1

RUN pip uninstall vllm -y
RUN pip install librosa
RUN git clone -b step-audio2-mini --depth 1 https://github.com/stepfun-ai/vllm.git /tmp/vllm \
    && MAX_JOBS=2 pip install -v /tmp/vllm \
    && rm -rf /tmp/vllm
ENTRYPOINT [""]
