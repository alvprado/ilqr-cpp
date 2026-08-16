FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libeigen3-dev \
    libgtest-dev \
    python3 python3-matplotlib python3-numpy python3-pil \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v1.1.2 https://github.com/autodiff/autodiff.git /tmp/autodiff \
    && cmake -S /tmp/autodiff -B /tmp/autodiff/build -DCMAKE_BUILD_TYPE=Release \
    -DAUTODIFF_BUILD_TESTS=OFF \
    -DAUTODIFF_BUILD_PYTHON=OFF \
    -DAUTODIFF_BUILD_EXAMPLES=OFF \
    -DAUTODIFF_BUILD_DOCS=OFF \
    && cmake --build /tmp/autodiff/build --target install \
    && rm -rf /tmp/autodiff

ENV MPLBACKEND=Agg

WORKDIR /workspace

COPY . . 

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DILQR_BUILD_EXAMPLES=ON \
    && cmake --build build -j

RUN chmod +x /workspace/docker_entrypoint.sh
ENTRYPOINT ["/workspace/docker_entrypoint.sh"]
CMD []