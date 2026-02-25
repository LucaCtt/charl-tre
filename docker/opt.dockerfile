FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# The uv installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run uv the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Copy the lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install required dependencies.
# The `--no-install-project` flag is used to avoid installing the project itself,
# which is handled in a later step to avoid re-downloading dependencies
# when the source code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src src
COPY README.md .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "opt"]