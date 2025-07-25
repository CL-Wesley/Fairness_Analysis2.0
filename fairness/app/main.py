from fastapi import FastAPI
from services.fairness.app.routers import fairness

app = FastAPI(
    title="Welcome to Fairness Service",
    description="Service for assessing and ensuring fairness in machine learning models.",
    version="3.1.0",
    docs_url="/fairness/docs",
    openapi_url="/fairness/openapi.json",
    redoc_url="/fairness/redocs"
)

from fastapi.middleware.cors import CORSMiddleware

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fairness/health", status_code=200)
def health_check():
    return {"status": "Fairness-healthy"}

app.include_router(fairness.routers)
