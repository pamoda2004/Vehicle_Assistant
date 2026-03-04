from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.recommender import recommend

app = FastAPI(title="AI Vehicle Buying Assistant")

# ✅ CORS enable (frontend -> backend calls allow)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # development stage
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    query: str
    top_n: int = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_api(req: Req):
    return recommend(req.query, req.top_n)

@app.get("/")
def home():
    return {"message": "AI Vehicle Buying Assistant API is running", "docs": "/docs"}