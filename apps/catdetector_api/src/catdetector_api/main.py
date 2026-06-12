from fastapi import FastAPI

from catdetector_api.routes import router

app = FastAPI(title="Cat Detector API")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run("catdetector_api.main:app", host="0.0.0.0", port=8000, reload=True)
