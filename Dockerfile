FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ARG INSTALL_TORCH=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt \
    && if [ "${INSTALL_TORCH}" = "1" ]; then python -m pip install torch torchvision; fi

COPY . /workspace

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
