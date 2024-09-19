from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Generator
import json
import time
import asyncio
import utils
import traceback

from utils import safe_decode
import config

app = FastAPI()
models = utils.load_models()

# Add CORS middleware. Allowing null for development use. TODO: based on env remove it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=16)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=50)
    repetition_penalty: float = Field(default=1.0)
    n: int = Field(default=1)
    stream: bool = Field(default=False)
    logprobs: int = Field(default=None)
    stop: Union[str, List[str]] = Field(default=None)

# Pydantic model for chat completions
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=16)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=50)
    repetition_penalty: float = Field(default=1.0)
    stream: bool = Field(default=False)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        model_name = request.model
        messages = request.messages
        max_new_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        top_k = request.top_k
        repetition_penalty = request.repetition_penalty
        stream = request.stream

        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_info = models[model_name]
        model, tokenizer, backend_config = model_info

        if not backend_config.public_api:
            raise HTTPException(status_code=403, detail=f"Access to model '{model_name}' is denied")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # or use: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Convert the chat messages to a single input string (concatenating the messages)
        conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        # Include attention_mask and pad_token_id
        inputs = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        n_input_tokens = input_ids.shape[1]
        max_length = n_input_tokens + max_new_tokens

        if stream:
            return StreamingResponse(
                stream_generate_chat(input_ids, attention_mask, model, tokenizer, None, True, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens),
                media_type='text/event-stream'
            )
        else:
            output_text = await generate_text(input_ids, attention_mask, model, tokenizer, True, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens)
            output_text = output_text.replace("|begin_of_text|>", "")
            response = create_chat_completion_response(output_text, model_name)
            return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

def create_chat_completion_response(text: str, model_name: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # You may want to calculate this
            "completion_tokens": 0,  # You may want to calculate this
            "total_tokens": 0  # You may want to calculate this
        }
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    try:
        model_name = request.model
        inputs = request.prompt
        max_new_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        top_k = request.top_k
        repetition_penalty = request.repetition_penalty
        stream = request.stream

        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_info = models[model_name]
        model, tokenizer, backend_config = model_info

        if not backend_config.public_api:
            raise HTTPException(status_code=403, detail=f"Access to model '{model_name}' is denied")

        if isinstance(inputs, list):
            inputs = inputs[0]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # or use: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    

        # Include attention_mask and pad_token_id
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        n_input_tokens = input_ids.shape[1]
        max_length = n_input_tokens + max_new_tokens

        if stream:
            return StreamingResponse(
                stream_generate(input_ids, attention_mask, model, tokenizer, request.stop, True, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens),
                media_type='text/event-stream'
            )
        else:
            output_text = await generate_text(input_ids, attention_mask, model, tokenizer, True, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens)
            output_text = output_text.replace("|begin_of_text|>", "")
            response = create_completion_response(output_text, model_name)
            return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        # Print the stack trace
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

def create_completion_response(text: str, model_name: str) -> dict:
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # You may want to calculate this
            "completion_tokens": 0,  # You may want to calculate this
            "total_tokens": 0  # You may want to calculate this
        }
    }

async def stream_generate(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens) -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0
        delta_q = []
        stop = False
        first_step = True
        start_time = int(time.time())
        
        while not stop:
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,  # Pass the attention mask here
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=1,
                session=session
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)

            if "\ufffd" in token_text:
                delta_q += delta
                continue

            delta_q = []
            all_outputs += token_text
            generated_tokens += 1

            response = {
                "id": f"cmpl-{start_time}",
                "object": "text_completion",
                "created": start_time,
                "model": model.__class__.__name__,
                "choices": [
                    {
                        "text": token_text.replace("|begin_of_text|>", ""),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(response)}\n\n"
            await asyncio.sleep(0)

            if stop_sequences and any(all_outputs.endswith(seq) for seq in stop_sequences):
                stop = True

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True

        yield "data: [DONE]\n\n"

async def stream_generate_chat(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens) -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0  
        delta_q = []
        stop = False
        first_step = True
        start_time = int(time.time())

        while not stop:
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,  # Pass the attention mask here
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=1,
                session=session
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)
            generated_tokens += 1

            if "\ufffd" in token_text:  # Skip undecodable token
                delta_q += delta
                continue

            delta_q = []
            all_outputs += token_text

            # Adjust the response to match the chat completion format
            response = {
                "id": f"chatcmpl-{start_time}",
                "object": "chat.completion",
                "created": start_time,
                "model": model.__class__.__name__,
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": token_text.replace("|begin_of_text|>", "")
                        },
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }

            # Stream the response as event data
            yield f"data: {json.dumps(response)}\n\n"
            await asyncio.sleep(0)

            # Handle stop sequences and termination
            if stop_sequences and any(all_outputs.endswith(seq) for seq in stop_sequences):
                stop = True

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True

        # Signal end of streaming
        yield "data: [DONE]\n\n"


async def generate_text(input_ids, attention_mask, model, tokenizer, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens) -> str:
    n_input_tokens = input_ids.shape[1]
 
    with model.inference_session(max_length=max_length) as session:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass the attention mask here
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            session=session
        )
        generated_tokens = outputs[0, n_input_tokens:].tolist()
        generated_text = safe_decode(tokenizer, generated_tokens)
        return generated_text

@app.get("/v1/models")
async def list_models():
    models_list = [
        {
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "organization-owner",
            "permission": [],
            "root": model_name,
            "parent": None
        }
        for model_name in models.keys()
    ]
    return {"object": "list", "data": models_list}
