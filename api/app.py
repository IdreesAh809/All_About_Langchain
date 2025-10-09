from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
import uvicorn

# Define input schema
class TopicRequest(BaseModel):
    topic: str

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Ollama Server",
    version="1.0",
    description="FastAPI API using Llama3 for essays and Llama2 for poems"
)

# Initialize models
llama3 = Ollama(model="llama3")
llama2 = Ollama(model="llama2")

# ----- Essay endpoint (Llama3) -----
@app.post("/essay")
async def generate_essay(request: TopicRequest):
    prompt = f"Write me an essay about {request.topic} in 100 words."
    response = llama3.invoke(prompt)
    return {"model": "llama3", "topic": request.topic, "essay": response}

# ----- Poem endpoint (Llama2) -----
@app.post("/poem")
async def generate_poem(request: TopicRequest):
    prompt = f"Write me a poem about {request.topic} for a 5-year-old child with 100 words."
    response = llama2.invoke(prompt)
    return {"model": "llama2", "topic": request.topic, "poem": response}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
