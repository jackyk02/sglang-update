from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
import numpy as np
import sglang as sgl
from token2action import TokenActionConverter, image_qa
import json_numpy as json
from typing import List, Optional, Union
import argparse

app = FastAPI()
converter = TokenActionConverter()

class BatchRequest(BaseModel):
    instructions: Union[List[str], str]  # Accept either a list of instructions or a single instruction
    image_path: str
    temperature: Optional[float] = 1.0

def process_batch(instructions: List[str], image_path: str, temperature: float):
    """Run a batch of inference with different instructions and return output token IDs and actions."""
    prompts = []
    for instruction in instructions:
        prompts.append({
            "image_path": image_path,
            "question": f"In: What action should the robot take to {instruction}?\nOut:"
        })

    states = image_qa.run_batch(prompts, max_new_tokens=7, temperature=temperature)
    
    output_ids = [np.array(s.get_meta_info("action")["output_ids"]) for s in states]
    actions = [np.array(converter.token_to_action(ids)) for ids in output_ids]
    return output_ids, actions

@app.get("/")
async def root():
    return {"message": "Batch processing server is running"}

@app.post("/batch")
async def handle_batch(request: Request):
    try:
        data = json.loads(await request.body())

        instructions = data.get("instructions")
        image_path = data.get("image_path")
        
        # Handle both single instruction and list of instructions
        if isinstance(instructions, str):
            instructions = [instructions]
        elif isinstance(instructions, list):
            if not all(isinstance(inst, str) for inst in instructions):
                raise HTTPException(status_code=400, detail="All instructions must be strings")
        else:
            raise HTTPException(status_code=400, detail="'instructions' must be a string or list of strings")
        
        if not isinstance(image_path, str):
            raise HTTPException(status_code=400, detail="'image_path' must be a string")

        temperature = float(data.get("temperature", 1.0))

        output_ids, actions = process_batch(instructions, image_path, temperature)
        return Response(content=json.dumps({"output_ids": output_ids, "actions": actions}),
                        media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=args.seed,
        trust_remote_code=True,
    )
    sgl.set_default_backend(runtime)
    uvicorn.run(app, host="0.0.0.0", port=3200)