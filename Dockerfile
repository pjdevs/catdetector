# syntax=docker/dockerfile:1

FROM node:24-bookworm-slim AS web-build

WORKDIR /app/apps/catdetector-web
COPY apps/catdetector-web/package.json apps/catdetector-web/package-lock.json ./
RUN npm ci
COPY apps/catdetector-web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

WORKDIR /app

ENV CATDETECTOR_WEB_DIST=/app/web
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY pyproject.toml uv.lock .python-version ./
COPY apps/catdetector-api/ apps/catdetector-api/
COPY apps/catdetector-trainer/pyproject.toml apps/catdetector-trainer/pyproject.toml
COPY packages/catdetector-model/ packages/catdetector-model/
RUN uv sync --locked --no-dev --package catdetector-api

COPY --from=web-build /app/apps/catdetector-web/dist/ /app/web/
COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

CMD ["uvicorn", "catdetector_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
