"""FastAPI placeholder for future deployment."""

from fastapi import FastAPI

app = FastAPI(title="Fruit Classifier API")


@app.get("/health")
def health_check():
    """Health endpoint."""
    return {"status": "ok", "service": "fruit-classifier"}
