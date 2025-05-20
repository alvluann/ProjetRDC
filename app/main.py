from fastapi import FastAPI
from construction_cbr import ConstructionCBREngine

engine = ConstructionCBREngine(
    similarity=ConstructionCBREngine.similarity,
    persistence_path="data/construction_cases.json")

app = FastAPI(title="IA Recomenda Materiais")

@app.post("/recommend")
async def recommend(problem: dict):
    solution, similarity = engine.cycle(problem)
    return {"similarity": similarity, **solution}
