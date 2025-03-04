from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Generator
from contextlib import asynccontextmanager
import json
import time
import asyncio
import utils
import traceback
import os
import signal
import sys

from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from utils import safe_decode
import config

# Add this to your config.py if not already there
if not hasattr(config, 'STREAM_BUFFER_SIZE'):
    config.STREAM_BUFFER_SIZE = 5  # Configurable token buffer size

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: models are already loaded before the app starts
    print("Application startup complete")
    yield
    # Shutdown: unload models when the app shuts down
    print("Application shutdown initiated, unloading models...")
    utils.unload_models(models)
    print("Models unloaded successfully")

app = FastAPI(lifespan=lifespan)
models = utils.load_models()

# Add CORS middleware. Allowing null for development use. TODO: based on env remove it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)
    repetition_penalty: float = Field(default=1.1)
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
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)
    repetition_penalty: float = Field(default=1.1)
    stream: bool = Field(default=False)
    stop: Union[str, List[str]] = Field(default=None)

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_tokens, tokenizer):
        self.stop_tokens = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]

    def __call__(self, input_ids, scores):
        return any(input_ids[0][-1] == stop_token for stop_token in self.stop_tokens)
    
def get_special_tokens(model_name: str):
    """
    Gets all special tokens for a Hugging Face model.
    
    :param model_name: The name or path of the model to load the tokenizer.
    :return: A set of special tokens for the model.
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Special tokens set - initialize with our problematic token
        special_tokens = {"|begin_of_text|>"}
        
        # We only need to remove the problematic token, not all special tokens
        # because many are needed for proper streaming
        
        return special_tokens
        
    except Exception as e:
        print(f"Error detecting special tokens for {model_name}: {str(e)}")
        # Return at least our problematic token if we can't get others
        return {"|begin_of_text|>"}

def clean_special_tokens(text: str, special_tokens: set) -> str:
    """
    Removes specified special tokens from the generated text.
    
    :param text: The text to clean.
    :param special_tokens: Set of special tokens to remove.
    :return: Cleaned text.
    """
    if not special_tokens:
        return text
    
    # Define tokens that should be removed from output
    tokens_to_remove = ["|begin_of_text|>"]
    
    # Add them to the special tokens set
    special_tokens.update(tokens_to_remove)
    
    cleaned = text
    # Sort tokens by length (longest first) to avoid substring issues
    for token in sorted(special_tokens, key=len, reverse=True):
        if token and token.strip():  # Avoid empty tokens
            cleaned = cleaned.replace(token, "")
    
    # Log token cleaning for debugging
    #if cleaned != text:
    #    print(f"Cleaned special tokens. Original length: {len(text)}, New length: {len(cleaned)}")
    
    return cleaned

def clean_role_markers(text: str) -> str:
    """
    Removes role markers and XML tags from the text.
    Standard cleaning for OpenAI-compatible API that focuses only on
    removing technical artifacts, not altering semantic content.
    
    :param text: The text to clean.
    :return: Cleaned text.
    """
    import re
    
    # Clean role prefix markers
    role_markers = [
        "assistant:", "Assistant:", "ASSISTANT:", 
        "user:", "User:", "USER:", 
        "system:", "System:", "SYSTEM:"
    ]
    
    # If the text starts with any of these markers, remove them
    cleaned_text = text
    for marker in role_markers:
        if cleaned_text.lstrip().startswith(marker):
            cleaned_text = cleaned_text.lstrip().replace(marker, "", 1).lstrip()
            print(f"Removed role marker: {marker}")
    
    return cleaned_text

def get_stop_tokens(model_name: str):
    """
    Detects stop sequences (EOS tokens, BOS tokens, and chat formatting rules) for a Hugging Face model.

    :param model_name: The name or path of the model to load the tokenizer.
    :return: A list of stop tokens for the model.
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Stop tokens list
        stop_tokens = set()

        # Detect EOS token
        if tokenizer.eos_token:
            stop_tokens.add(tokenizer.eos_token)

        # Detect BOS token (not necessarily a stop token but useful for parsing)
        if tokenizer.bos_token:
            stop_tokens.add(tokenizer.bos_token)

        # Detect additional special tokens
        if tokenizer.additional_special_tokens:
            stop_tokens.update(tokenizer.additional_special_tokens)

        # Try to locate the tokenizer.json file
        tokenizer_json_path = tokenizer.init_kwargs.get("tokenizer_file")

        if tokenizer_json_path and os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

            # Extract BOS and EOS tokens
            if "bos_token" in tokenizer_config and isinstance(tokenizer_config["bos_token"], dict):
                stop_tokens.add(tokenizer_config["bos_token"].get("content", ""))

            if "eos_token" in tokenizer_config and isinstance(tokenizer_config["eos_token"], dict):
                stop_tokens.add(tokenizer_config["eos_token"].get("content", ""))

            # Extract special tokens from `added_tokens_decoder`
            if "added_tokens_decoder" in tokenizer_config:
                for token_info in tokenizer_config["added_tokens_decoder"].values():
                    content = token_info.get("content")
                    if content and token_info.get("special", False):
                        # Avoid adding reserved special tokens
                        if not content.startswith("<|reserved_special_token_"):
                            stop_tokens.add(content)

            # Handle chat template special cases
            if "chat_template" in tokenizer_config:
                chat_template = tokenizer_config["chat_template"]

                # Detect structured message patterns
                if "<|eot_id|>" in chat_template:
                    stop_tokens.add("<|eot_id|>")  # Assistant response terminator
                
                if "<|end_of_text|>" in chat_template:
                    stop_tokens.add("<|end_of_text|>")  # Another valid EOS marker

                if "<|start_header_id|>" in chat_template and "<|end_header_id|>" in chat_template:
                    stop_tokens.add("<|end_header_id|>")  # End of structured response

        return sorted(stop_tokens) if stop_tokens else None

    except Exception as e:
        traceback.print_exc()
        print(f"Error detecting stop tokens for {model_name}: {str(e)}")
        return None

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

        #print("Incoming request:", request.model_dump_json(indent=2))

        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_info = models[model_name]
        model, tokenizer, backend_config = model_info

        if not backend_config.public_api:
            raise HTTPException(status_code=403, detail=f"Access to model '{model_name}' is denied")

        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        # Check for system message
        system_content = ""
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break
                
        # Format the conversation messages in a way the model can understand
        conversation_history = ""
        
        # Add system message if present
        if system_content:
            conversation_history += f"system: {system_content}\n\n"
            
        # Add user/assistant messages
        for msg in messages:
            if msg.role != "system":  # Skip system messages as we already added them
                conversation_history += f"{msg.role}: {msg.content}\n"
        
        # Add the assistant role marker at the end to prompt the model for its response
        conversation_history += "assistant:"

        inputs = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        max_length = input_ids.shape[1] + max_new_tokens
        
        # Store the prompt token count
        prompt_tokens = input_ids.shape[1]

        # Get model-specific stop tokens
        detected_stop_tokens = get_stop_tokens(model_name)

        # Merge user-provided stop tokens if any
        stop_sequences = detected_stop_tokens or []
        if request.stop:
            if isinstance(request.stop, list):
                stop_sequences.extend(request.stop)
            else:
                stop_sequences.append(request.stop)
                
        # Add common chat patterns as stop sequences
        chat_stop_sequences = [
            "\nuser:", "\nassistant:", "\nsystem:",
            "<user>", "</user>", "<assistant>", "</assistant>", 
            "<system>", "</system>"
        ]
        stop_sequences.extend(chat_stop_sequences)

        if stream:
            return StreamingResponse(
                stream_generate_chat(
                    input_ids, attention_mask, model, tokenizer, 
                    stop_sequences, True, temperature, top_p, top_k, 
                    repetition_penalty, max_length, max_new_tokens, prompt_tokens
                ),
                media_type='text/event-stream'
            )
        else:
            output_text, prompt_token_count, completion_token_count = await generate_text(
                input_ids, attention_mask, model, tokenizer, 
                stop_sequences, True, temperature, top_p, top_k, 
                repetition_penalty, max_length, max_new_tokens
            )            
            response = create_chat_completion_response(output_text, model_name, prompt_token_count, completion_token_count)
            return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

        #print("Incoming request:", request.model_dump_json(indent=2))

        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_info = models[model_name]
        model, tokenizer, backend_config = model_info

        if not backend_config.public_api:
            raise HTTPException(status_code=403, detail=f"Access to model '{model_name}' is denied")

        if isinstance(inputs, list):
            inputs = inputs[0]

        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        max_length = input_ids.shape[1] + max_new_tokens
        
        # Store the prompt token count
        prompt_tokens = input_ids.shape[1]
        print(f"Token count methods - shape[1]: {input_ids.shape[1]}, numel: {input_ids.numel()}, encoded length: {len(tokenizer.encode(inputs))}")


        # Get model-specific stop tokens
        detected_stop_tokens = get_stop_tokens(model_name)

        # Merge user-provided stop tokens if any
        stop_sequences = detected_stop_tokens or []
        if request.stop:
            if isinstance(request.stop, list):
                stop_sequences.extend(request.stop)
            else:
                stop_sequences.append(request.stop)

        if stream:
            return StreamingResponse(
                stream_generate(
                    input_ids, attention_mask, model, tokenizer, 
                    stop_sequences, True, temperature, top_p, top_k, 
                    repetition_penalty, max_length, max_new_tokens, prompt_tokens
                ),
                media_type='text/event-stream'
            )
        else:
            output_text, prompt_token_count, completion_token_count = await generate_text(
                input_ids, attention_mask, model, tokenizer, 
                stop_sequences, True, temperature, top_p, top_k, 
                repetition_penalty, max_length, max_new_tokens
            )
            response = create_completion_response(output_text, model_name, prompt_token_count, completion_token_count)
            return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


def create_chat_completion_response(text: str, model_name: str, prompt_tokens: int, completion_tokens: int) -> dict:
    # Final cleanup of any XML tags that might still be present
    cleaned_text = clean_role_markers(text)
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": cleaned_text
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def create_completion_response(text: str, model_name: str, prompt_tokens: int, completion_tokens: int) -> dict:
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

async def stream_generate(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens, prompt_tokens) -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    
    # Get special tokens to remove - just the ones we want to hide from users
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    
    # Use the buffer size from config
    buffer_size = config.STREAM_BUFFER_SIZE
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0
        delta_q = []
        # Initialize token buffer
        token_buffer = []
        # We'll maintain both raw and cleaned buffer text
        raw_buffer_text = ""
        cleaned_buffer_text = ""
        
        stop = False
        stopped_due_to_sequence = False
        stopping_sequence = None
        first_step = True
        start_time = int(time.time())
        
        while not stop:
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=1,
                session=session,
                pad_token_id=tokenizer.pad_token_id
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)

            if "\ufffd" in token_text:
                delta_q += delta
                continue

            delta_q = []
            
            # Add new token to all_outputs for overall tracking
            all_outputs += token_text
            generated_tokens += 1
            #print(f"Generated token: {token_text!r}")
            
            # Clean the token text before adding to buffer
            cleaned_token = clean_special_tokens(token_text, special_tokens)
            
            # Add current token to buffer (both raw and cleaned versions)
            token_buffer.append((delta, token_text, cleaned_token))
            raw_buffer_text += token_text
            cleaned_buffer_text += cleaned_token
            
            # Check if any stop sequence is in the cleaned buffer
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_buffer_text:
                    # Found a stop sequence
                    stop_idx = cleaned_buffer_text.find(stop_seq)
                    
                    # Calculate how many tokens to keep (those before the stop sequence)
                    current_pos = 0
                    tokens_to_keep = []
                    for i, (d, raw, clean) in enumerate(token_buffer):
                        if current_pos + len(clean) > stop_idx:
                            # This token contains the beginning of the stop sequence
                            if current_pos < stop_idx:
                                # Keep the part before the stop sequence
                                keep_chars = stop_idx - current_pos
                                if keep_chars > 0:
                                    partial_clean = clean[:keep_chars]
                                    if partial_clean:  # Only add if there's content
                                        tokens_to_keep.append((d, raw, partial_clean))
                            break
                        else:
                            tokens_to_keep.append((d, raw, clean))
                            current_pos += len(clean)
                    
                    # Update buffer with only the tokens to keep
                    token_buffer = tokens_to_keep
                    
                    # Mark as stopped due to sequence
                    stopped_due_to_sequence = True
                    stopping_sequence = stop_seq
                    
                    # Recalculate buffer texts
                    raw_buffer_text = "".join(t[1] for t in token_buffer)
                    cleaned_buffer_text = "".join(t[2] for t in token_buffer)
                    
                    stop_found = True
                    stop = True
                    break
            
            # Stream tokens from buffer when:
            # 1. Buffer reaches configured size
            # 2. A stop sequence was found
            # 3. We've reached max tokens
            if len(token_buffer) >= buffer_size or stop_found or generated_tokens == max_new_tokens:
                # Always send at least the first token if buffer has content
                while token_buffer and (len(token_buffer) > 1 or stop):
                    # Get the first token from buffer (already cleaned)
                    _, _, first_cleaned_token = token_buffer.pop(0)
                    
                    # Only send if there's actual content
                    if first_cleaned_token:
                        # Add usage statistics only on the final message
                        usage = None
                        if stop and not token_buffer:
                            usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": generated_tokens,
                                "total_tokens": prompt_tokens + generated_tokens
                            }
                        
                        response = {
                            "id": f"cmpl-{start_time}",
                            "object": "text_completion",
                            "created": start_time,
                            "model": model.__class__.__name__,
                            "choices": [
                                {
                                    "text": first_cleaned_token,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None if not stop else "stop"
                                }
                            ]
                        }
                        
                        # Add usage if this is the final message
                        if usage:
                            response["usage"] = usage
                        
                        yield f"data: {json.dumps(response)}\n\n"
                        await asyncio.sleep(0)
                
                # Update buffer texts by removing the sent tokens
                raw_buffer_text = "".join(t[1] for t in token_buffer)
                cleaned_buffer_text = "".join(t[2] for t in token_buffer)

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True
                
        # If we stopped due to a stop sequence, the buffer should be empty or contain only one token
        # We only flush the last token if we're not stopped due to a sequence
        if token_buffer and not stopped_due_to_sequence:
            _, _, last_cleaned_token = token_buffer[0]
            if last_cleaned_token:
                # Include usage statistics in the final response
                response = {
                    "id": f"cmpl-{start_time}",
                    "object": "text_completion",
                    "created": start_time,
                    "model": model.__class__.__name__,
                    "choices": [
                        {
                            "text": last_cleaned_token,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated_tokens,
                        "total_tokens": prompt_tokens + generated_tokens
                    }
                }
                
                yield f"data: {json.dumps(response)}\n\n"
                await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

async def stream_generate_chat(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens, prompt_tokens) -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    
    # Get special tokens to remove - just the ones we want to hide from users
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    
    # Use the buffer size from config
    buffer_size = config.STREAM_BUFFER_SIZE
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0  
        delta_q = []
        
        # Initialize token buffer
        token_buffer = []
        # We'll maintain both raw and cleaned buffer text
        raw_buffer_text = ""
        cleaned_buffer_text = ""
        
        stop = False
        stopped_due_to_sequence = False
        stopping_sequence = None
        first_step = True
        start_time = int(time.time())
        # Track if we need to clean initial role marker
        initial_token = True

        while not stop:
            outputs = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=1,
                session=session,
                pad_token_id=tokenizer.pad_token_id
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)
            generated_tokens += 1
            #print(f"Generated token: {token_text!r}")

            if "\ufffd" in token_text:  # Skip undecodable token
                delta_q += delta
                continue

            delta_q = []
            
            # Add new token to all_outputs for overall tracking
            all_outputs += token_text
            
            # Clean the token text
            cleaned_token = clean_special_tokens(token_text, special_tokens)
            
            # Clean role markers if this is the first token
            if initial_token:
                cleaned_token = clean_role_markers(cleaned_token)
                initial_token = False
                
            # Add current token to buffer (both raw and cleaned versions)
            token_buffer.append((delta, token_text, cleaned_token))
            raw_buffer_text += token_text
            cleaned_buffer_text += cleaned_token
            
            # Check if any stop sequence is in the cleaned buffer
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_buffer_text:
                    # Found a stop sequence
                    stop_idx = cleaned_buffer_text.find(stop_seq)
                    
                    # Calculate how many tokens to keep (those before the stop sequence)
                    current_pos = 0
                    tokens_to_keep = []
                    for i, (d, raw, clean) in enumerate(token_buffer):
                        if current_pos + len(clean) > stop_idx:
                            # This token contains the beginning of the stop sequence
                            if current_pos < stop_idx:
                                # Keep the part before the stop sequence
                                keep_chars = stop_idx - current_pos
                                if keep_chars > 0:
                                    partial_clean = clean[:keep_chars]
                                    if partial_clean:  # Only add if there's content
                                        tokens_to_keep.append((d, raw, partial_clean))
                            break
                        else:
                            tokens_to_keep.append((d, raw, clean))
                            current_pos += len(clean)
                    
                    # Update buffer with only the tokens to keep
                    token_buffer = tokens_to_keep
                    
                    # Mark as stopped due to sequence
                    stopped_due_to_sequence = True
                    stopping_sequence = stop_seq
                    
                    # Recalculate buffer texts
                    raw_buffer_text = "".join(t[1] for t in token_buffer)
                    cleaned_buffer_text = "".join(t[2] for t in token_buffer)
                    
                    stop_found = True
                    stop = True
                    break
            
            # Stream tokens from buffer when:
            # 1. Buffer reaches configured size
            # 2. A stop sequence was found
            # 3. We've reached max tokens
            if len(token_buffer) >= buffer_size or stop_found or generated_tokens == max_new_tokens:
                # Always send at least the first token if buffer has content
                while token_buffer and (len(token_buffer) > 1 or stop):
                    # Get the first token from buffer (already cleaned)
                    _, _, first_cleaned_token = token_buffer.pop(0)
                    
                    # Only send if there's actual content
                    if first_cleaned_token:
                        # Add usage statistics only on the final message
                        usage = None
                        if stop and not token_buffer:
                            usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": generated_tokens,
                                "total_tokens": prompt_tokens + generated_tokens
                            }
                        
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
                                        "content": first_cleaned_token
                                    },
                                    "index": 0,
                                    "finish_reason": None if not stop else "stop"
                                }
                            ]
                        }
                        
                        # Add usage if this is the final message
                        if usage:
                            response["usage"] = usage
                        
                        # Stream the response as event data
                        yield f"data: {json.dumps(response)}\n\n"
                        await asyncio.sleep(0)
                
                # Update buffer texts by removing the sent tokens
                raw_buffer_text = "".join(t[1] for t in token_buffer)
                cleaned_buffer_text = "".join(t[2] for t in token_buffer)

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True
                
        # If we stopped due to a stop sequence, the buffer should be empty or contain only one token
        # We only flush the last token if we're not stopped due to a sequence
        if token_buffer and not stopped_due_to_sequence:
            _, _, last_cleaned_token = token_buffer[0]
            if last_cleaned_token:
                response = {
                    "id": f"chatcmpl-{start_time}",
                    "object": "chat.completion",
                    "created": start_time,
                    "model": model.__class__.__name__,
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": last_cleaned_token
                            },
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": generated_tokens,
                        "total_tokens": prompt_tokens + generated_tokens
                    }
                }
                
                yield f"data: {json.dumps(response)}\n\n"
                await asyncio.sleep(0)

        # Signal end of streaming
        yield "data: [DONE]\n\n"

async def generate_text(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens) -> tuple:
    prompt_tokens = input_ids.shape[1]
    n_input_tokens = input_ids.shape[1]
    all_outputs = ""
    delta_q = []
    stop = False
    first_step = True
    generated_tokens = 0
    
    # Get special tokens to remove - just the ones we want to hide from users
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    
    with model.inference_session(max_length=max_length) as session:
        while not stop:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=1,
                session=session,
                pad_token_id=tokenizer.pad_token_id
            )
            
            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)
            generated_tokens += 1
            #print(f"Generated token: {token_text!r}")

            # Clean the token for comparison with stop sequences
            cleaned_token = clean_special_tokens(token_text, special_tokens)
            
            # Check if adding this token would create a stop sequence
            potential_text = all_outputs + token_text
            cleaned_potential = clean_special_tokens(potential_text, special_tokens)
            
            # Check for stop sequences in cleaned potential text
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_potential:
                    # Found a stop sequence, remove it and everything after
                    stop_idx = cleaned_potential.find(stop_seq)
                    # Only keep text before the stop sequence
                    all_outputs = potential_text[:len(potential_text) - len(token_text) + (stop_idx - len(cleaned_potential) + len(token_text))]
                    print(f"Got stop seq: {stop_seq!r}, truncating output")
                    stop_found = True
                    stop = True
                    break
            
            if not stop_found:
                # If no stop sequence, add the token to all_outputs
                if "\ufffd" in token_text:  # Skip undecodable token
                    delta_q += delta
                    continue
                
                delta_q = []
                all_outputs += token_text
            
            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False
            
            if generated_tokens == max_new_tokens: 
                stop = True
                print("Reached max tokens")                
    
    # Clean special tokens before returning
    cleaned_output = clean_special_tokens(all_outputs, special_tokens)
    #print(f"Final output length before cleaning: {len(all_outputs)}, after cleaning: {len(cleaned_output)}")
    return cleaned_output, prompt_tokens, generated_tokens


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


# Import for process control
import subprocess
import threading

# Running server reference
server_process = None

# Signal handlers for clean shutdown
def handle_sigterm(signum, frame):
    print("Received SIGTERM. Initiating graceful shutdown...")
    # Manual model unload before termination
    utils.unload_models(models)
    print("Models unloaded successfully")
    # Now exit gracefully
    sys.exit(0)

def handle_sigint(signum, frame):
    print("Received SIGINT. Initiating graceful shutdown...")
    # Manual model unload before termination
    utils.unload_models(models)
    print("Models unloaded successfully")
    # Now exit gracefully
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigint)