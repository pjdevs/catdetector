# syntax=docker/dockerfile:1

FROM node:24-bookworm-slim AS web-build

WORKDIR /app/apps/catdetector-web
COPY apps/catdetector-web/package.json apps/catdetector-web/package-lock.json ./
RUN npm ci
COPY apps/catdetector-web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

WORKDIR /app
ARG UV_TORCH_BACKEND=cu130

ENV CATDETECTOR_WEB_DIST=/app/web
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY apps/catdetector-api/ apps/catdetector-api/
COPY packages/catdetector-model/ packages/catdetector-model/
RUN uv venv /app/.venv \
    && uv pip install \
        --python /app/.venv/bin/python \
        --torch-backend "${UV_TORCH_BACKEND}" \
        "torch>=2.12.0" \
        "torchvision>=0.27.0" \
    && uv pip install \
        --python /app/.venv/bin/python \
        --no-sources \
        ./packages/catdetector-model \
        ./apps/catdetector-api

COPY --from=web-build /app/apps/catdetector-web/dist/ /app/web/
COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

CMD ["catdetector-api"]
