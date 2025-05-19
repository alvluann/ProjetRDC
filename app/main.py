from fastapi import FastAPI
from cbr_engine import CBREngine
engine = CBREngine(persistence_path="data/cases.json")
app = FastAPI()

@app.post("/solve")
async def solve(problem: dict):
    sol, sim = engine.cycle(problem)
    return {"solution": sol, "similarity": sim}
