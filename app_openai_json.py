from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Optional, Literal, Any, Dict, Generator
from contextlib import asynccontextmanager
import json
import time
import asyncio
import utils
import traceback
import os
import signal
import sys
import re
from functools import lru_cache
import uuid

from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from utils import safe_decode
import config

# Add this to your config.py if not already there
if not hasattr(config, 'STREAM_BUFFER_SIZE'):
    config.STREAM_BUFFER_SIZE = 5


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FunctionParameters(BaseModel):
    type: str = "object"
    properties: Dict[str, Any] = {}
    required: List[str] = []


class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class ToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, str]


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, str]


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


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
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], ToolChoice]] = "auto"


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


class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_tokens, tokenizer):
        self.stop_tokens = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]

    def __call__(self, input_ids, scores):
        return any(input_ids[0][-1] == stop_token for stop_token in self.stop_tokens)


# =============================================================================
# MODEL DETECTION AND TEMPLATE UTILITIES
# =============================================================================

def detect_model_family(model_name: str) -> str:
    """Detect the model family for template selection."""
    model_name_lower = model_name.lower()
    
    if "hermes" in model_name_lower:
        return "hermes"
    elif "llama" in model_name_lower and ("3.1" in model_name or "3.2" in model_name or "3.3" in model_name):
        return "llama3"
    elif "mistral" in model_name_lower and "nemo" in model_name_lower:
        return "mistral_nemo"
    elif "mistral" in model_name_lower:
        return "mistral"
    elif "command" in model_name_lower or "c4ai" in model_name_lower:
        return "command_r"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "functionary" in model_name_lower:
        return "functionary"
    else:
        return "generic"


def has_native_tool_template(tokenizer) -> bool:
    """Check if the tokenizer has native tool calling template support."""
    if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
        return False
    
    # Handle different chat_template formats
    template_content = ""
    if isinstance(tokenizer.chat_template, str):
        # String format (Jinja2 template)
        template_content = tokenizer.chat_template.lower()
    elif isinstance(tokenizer.chat_template, dict):
        # Dictionary format - convert to string for searching
        template_content = str(tokenizer.chat_template).lower()
    else:
        # Other formats - try to convert to string
        try:
            template_content = str(tokenizer.chat_template).lower()
        except Exception:
            return False
    
    tool_keywords = ['tool', 'function', 'tool_call', 'tool_calls']
    return any(keyword in template_content for keyword in tool_keywords)


def convert_tools_to_hf_format(tools: List[Tool]) -> List[Dict[str, Any]]:
    """Convert tools to HuggingFace format for native template usage."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters.dict()
            }
        }
        for tool in tools
    ]


# =============================================================================
# TEMPLATE FORMATTING FUNCTIONS
# =============================================================================

def format_tools_for_prompt(tools: List[Tool]) -> str:
    """Format tools as a string for generic prompt-based approach (fallback)."""
    if not tools:
        return ""
    
    tools_text = "\n\nYou have access to the following tools:\n"
    for tool in tools:
        func = tool.function
        tools_text += f"\n- {func.name}: {func.description}\n"
        if func.parameters.properties:
            tools_text += "  Parameters:\n"
            for param_name, param_info in func.parameters.properties.items():
                param_desc = param_info.get('description', 'No description')
                param_type = param_info.get('type', 'unknown')
                required = "(required)" if param_name in func.parameters.required else "(optional)"
                tools_text += f"    - {param_name} ({param_type}) {required}: {param_desc}\n"
    
    tools_text += "\nTo call a function, respond with a JSON object in this exact format:\n"
    tools_text += '{"tool_call": {"name": "function_name", "arguments": {"param1": "value1"}}}\n'
    tools_text += "Only use this format when you need to call a function. For regular responses, respond normally.\n"
    
    return tools_text


def format_hermes_tools_prompt(tools: List[Tool]) -> str:
    """Format tools for Hermes models using their specific format."""
    if not tools:
        return ""
    
    tools_xml = "\n<tools>\n"
    for tool in tools:
        func = tool.function
        tools_xml += f'<tool_description>\n'
        tools_xml += f'<tool_name>{func.name}</tool_name>\n'
        tools_xml += f'<description>{func.description}</description>\n'
        tools_xml += f'<parameters>\n{json.dumps(func.parameters.dict(), indent=2)}\n</parameters>\n'
        tools_xml += f'</tool_description>\n'
    tools_xml += "</tools>\n"
    
    tools_xml += "\nYou may call one or more functions to assist with the user query. "
    tools_xml += "Don't make assumptions about what values to plug into functions. "
    tools_xml += "Ask for clarification if a user request is ambiguous.\n"
    tools_xml += "For each function call return a json object with function name and arguments "
    tools_xml += "within <tool_call></tool_call> tags with the following schema:\n"
    tools_xml += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-dict>}\n</tool_call>\n"
    
    return tools_xml


def format_hermes_conversation(messages: List[ChatMessage], tools: List[Tool], tokenizer) -> str:
    """Format conversation for Hermes models using ChatML format."""
    conversation = ""
    
    # System message with tools
    system_content = ""
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
            break
    
    if tools:
        system_content += format_hermes_tools_prompt(tools)
    
    if system_content:
        conversation += f"<|im_start|>system\n{system_content}<|im_end|>\n"
    
    # Add other messages
    for msg in messages:
        if msg.role == "system":
            continue  # Already handled
        elif msg.role == "tool":
            conversation += f"<|im_start|>tool\n{msg.content}<|im_end|>\n"
        elif msg.tool_calls:
            conversation += f"<|im_start|>assistant\n"
            for tool_call in msg.tool_calls:
                conversation += f"<tool_call>\n{tool_call.function['arguments']}\n</tool_call>\n"
            conversation += f"<|im_end|>\n"
        else:
            conversation += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    
    conversation += "<|im_start|>assistant\n"
    return conversation


def format_llama3_conversation(messages: List[ChatMessage], tools: List[Tool], tokenizer) -> str:
    """Format conversation for Llama 3.1+ models."""
    try:
        if tools:
            hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages if msg.role != "tool"]
            hf_tools = convert_tools_to_hf_format(tools)
            
            return tokenizer.apply_chat_template(
                hf_messages,
                tools=hf_tools, 
                tokenize=False,
                add_generation_prompt=True
            )
    except Exception:
        pass
    
    return format_generic_conversation(messages, tools)


def format_mistral_conversation(messages: List[ChatMessage], tools: List[Tool], tokenizer) -> str:
    """Format conversation for Mistral models."""
    conversation = ""
    
    if tools:
        tools_json = json.dumps(convert_tools_to_hf_format(tools), indent=2)
        conversation += f"[AVAILABLE_TOOLS] {tools_json} [/AVAILABLE_TOOLS]\n\n"
    
    for msg in messages:
        if msg.role == "system":
            conversation += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "user":
            conversation += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "assistant":
            if msg.tool_calls:
                tool_calls_json = json.dumps([
                    {"name": tc.function["name"], "arguments": json.loads(tc.function["arguments"])}
                    for tc in msg.tool_calls
                ])
                conversation += f"[TOOL_CALLS] {tool_calls_json}\n"
            else:
                conversation += f"{msg.content}\n"
        elif msg.role == "tool":
            conversation += f"[TOOL_RESULTS] {msg.content} [/TOOL_RESULTS]\n"
    
    return conversation


def format_generic_conversation(messages: List[ChatMessage], tools: List[Tool]) -> str:
    """Generic conversation formatting (fallback)."""
    system_content = ""
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
            break
    
    conversation_history = ""
    
    if system_content:
        conversation_history += f"system: {system_content}\n\n"
    
    if tools:
        if system_content:
            conversation_history = conversation_history.rstrip() + "\n" + format_tools_for_prompt(tools) + "\n\n"
        else:
            conversation_history += format_tools_for_prompt(tools) + "\n\n"
        
    for msg in messages:
        if msg.role == "system":
            continue
        elif msg.role == "tool":
            conversation_history += f"tool_result: {msg.content}\n"
        else:
            conversation_history += f"{msg.role}: {msg.content}\n"
    
    conversation_history += "assistant:"
    return conversation_history


def format_conversation_with_tools(messages: List[ChatMessage], tools: List[Tool], tokenizer, model_name: str) -> str:
    """Format conversation with tools using the best available method for the model."""
    model_family = detect_model_family(model_name)
    
    # Try native HuggingFace template first
    if tools and has_native_tool_template(tokenizer):
        try:
            hf_messages = []
            for msg in messages:
                if msg.role == "tool":
                    hf_messages.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id
                    })
                elif msg.tool_calls:
                    hf_messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function", 
                                "function": {
                                    "name": tc.function["name"],
                                    "arguments": tc.function["arguments"]
                                }
                            }
                            for tc in msg.tool_calls
                        ]
                    })
                else:
                    hf_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            hf_tools = convert_tools_to_hf_format(tools)
            
            formatted = tokenizer.apply_chat_template(
                hf_messages,
                tools=hf_tools,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"Using native template for {model_name}")
            return formatted
            
        except Exception as e:
            print(f"Native template failed for {model_name}: {e}, falling back to model-specific")
    
    # Model-specific formatting
    if model_family == "hermes":
        return format_hermes_conversation(messages, tools, tokenizer)
    elif model_family == "llama3":
        return format_llama3_conversation(messages, tools, tokenizer) 
    elif model_family == "mistral" or model_family == "mistral_nemo":
        return format_mistral_conversation(messages, tools, tokenizer)
    else:
        return format_generic_conversation(messages, tools)


# =============================================================================
# TOOL CALL EXTRACTION FUNCTIONS (FIXED)
# =============================================================================

def extract_hermes_tool_call(text: str, tools: List[Tool]) -> Optional[ToolCall]:
    """Extract tool call from Hermes model response with enhanced patterns and incomplete tag handling."""
    
    def extract_balanced_json(text_after_tag: str) -> Optional[str]:
        """Extract a balanced JSON object, handling nested braces properly."""
        text_after_tag = text_after_tag.strip()
        if not text_after_tag.startswith('{'):
            return None
            
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text_after_tag):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text_after_tag[:i+1]
        
        # If we get here, JSON is incomplete
        return text_after_tag if brace_count > 0 else None
    
    # Try complete tool calls first
    complete_patterns = [
        r'<tool_call>\s*(.*?)\s*</tool_call>',  # Complete tool call with any content
    ]
    
    for pattern in complete_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                function_name = parsed.get("name")
                
                if function_name and any(tool.function.name == function_name for tool in tools):
                    print(f"🔧 Complete tool call found: {function_name}")
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function={
                            "name": function_name,
                            "arguments": json.dumps(parsed.get("arguments", {}))
                        }
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing complete tool call: {e}")
                continue
    
    # NEW: Handle incomplete tool calls (missing closing tag) - like Ollama
    # Look for opening tag and extract JSON manually
    tool_call_match = re.search(r'<tool_call>\s*(.*)', text, re.DOTALL)
    if tool_call_match:
        content_after_tag = tool_call_match.group(1)
        
        # Extract balanced JSON
        json_str = extract_balanced_json(content_after_tag)
        if json_str:
            try:
                # Try to parse the JSON as-is first
                parsed = json.loads(json_str)
                function_name = parsed.get("name")
                
                if function_name and any(tool.function.name == function_name for tool in tools):
                    print(f"🔧 Complete tool call found (no closing tag): {function_name}")
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function={
                            "name": function_name,
                            "arguments": json.dumps(parsed.get("arguments", {}))
                        }
                    )
            except json.JSONDecodeError:
                # Try to fix incomplete JSON
                fixed_json = json_str
                
                # Add missing closing braces
                if fixed_json.count('{') > fixed_json.count('}'):
                    missing_braces = fixed_json.count('{') - fixed_json.count('}')
                    fixed_json += '}' * missing_braces
                    print(f"🔧 Fixed incomplete JSON by adding {missing_braces} closing braces")
                
                # Try to fix incomplete quoted values
                if fixed_json.count('"') % 2 == 1:
                    fixed_json += '"'
                    print(f"🔧 Fixed incomplete JSON by adding closing quote")
                    
                # If still incomplete, try to close the object
                if not fixed_json.endswith('}'):
                    fixed_json += '}'
                    print(f"🔧 Fixed incomplete JSON by adding final closing brace")
                
                try:
                    parsed = json.loads(fixed_json)
                    function_name = parsed.get("name")
                    
                    if function_name and any(tool.function.name == function_name for tool in tools):
                        print(f"🔧 Incomplete tool call completed and found: {function_name}")
                        return ToolCall(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            type="function",
                            function={
                                "name": function_name,
                                "arguments": json.dumps(parsed.get("arguments", {}))
                            }
                        )
                except json.JSONDecodeError as e:
                    print(f"Error parsing fixed JSON: {e}")
    
    return None
    return None


def extract_llama3_tool_call(text: str, tools: List[Tool]) -> Optional[ToolCall]:
    """Extract tool call from Llama 3.1+ model response."""
    patterns = [
        r'<\|python_tag\|>\s*([^<]+)',
        r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}',
        r'```json\s*(\{.*?\})\s*```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                if 'python_tag' in pattern:
                    python_call = match.group(1).strip()
                    func_match = re.match(r'(\w+)\((.*)\)', python_call)
                    if func_match:
                        function_name = func_match.group(1)
                        args_str = func_match.group(2)
                        arguments = {}
                        if args_str:
                            for arg in args_str.split(','):
                                if '=' in arg:
                                    key, val = arg.split('=', 1)
                                    key = key.strip()
                                    val = val.strip().strip('"\'')
                                    arguments[key] = val
                else:
                    if pattern.startswith(r'\{'):
                        function_name = match.group(1)
                        arguments = json.loads(match.group(2))
                    else:
                        parsed = json.loads(match.group(1))
                        function_name = parsed.get("name")
                        arguments = parsed.get("arguments", {})
                
                if function_name and any(tool.function.name == function_name for tool in tools):
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function={
                            "name": function_name,
                            "arguments": json.dumps(arguments)
                        }
                    )
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing Llama3 tool call with pattern {pattern}: {e}")
                continue
    
    return None


def extract_mistral_tool_call(text: str, tools: List[Tool]) -> Optional[ToolCall]:
    """Extract tool call from Mistral model response."""
    pattern = r'\[TOOL_CALLS\]\s*(\[.*?\])'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(1)
            parsed = json.loads(json_str)
            
            if isinstance(parsed, list) and len(parsed) > 0:
                call_info = parsed[0]
                function_name = call_info.get("name")
                
                if function_name and any(tool.function.name == function_name for tool in tools):
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function={
                            "name": function_name,
                            "arguments": json.dumps(call_info.get("arguments", {}))
                        }
                    )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing Mistral tool call: {e}")
    
    return None


def extract_generic_tool_call(text: str, tools: List[Tool]) -> Optional[ToolCall]:
    """Extract tool call using generic patterns (fallback)."""
    patterns = [
        r'\{"tool_call":\s*\{[^}]*\}\s*\}',
        r'\{[^}]*"name"[^}]*"arguments"[^}]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                
                if "tool_call" in parsed:
                    call_info = parsed["tool_call"]
                else:
                    call_info = parsed
                
                if "name" in call_info:
                    function_name = call_info["name"]
                    if any(tool.function.name == function_name for tool in tools):
                        return ToolCall(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            type="function",
                            function={
                                "name": function_name,
                                "arguments": json.dumps(call_info.get("arguments", {}))
                            }
                        )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing generic tool call with pattern {pattern}: {e}")
                continue
    
    return None


def extract_tool_call_from_response(text: str, tools: List[Tool], model_name: str = "") -> Optional[ToolCall]:
    """Extract tool call from model response using model-specific patterns."""
    if not tools:
        return None
    
    model_family = detect_model_family(model_name)
    
    if model_family == "hermes":
        return extract_hermes_tool_call(text, tools)
    elif model_family == "llama3":
        return extract_llama3_tool_call(text, tools)
    elif model_family in ["mistral", "mistral_nemo"]:
        return extract_mistral_tool_call(text, tools)
    else:
        return extract_generic_tool_call(text, tools)


def is_tool_call_potentially_incomplete(text: str, model_family: str) -> bool:
    """Check if the text contains an incomplete tool call that might be completed with more tokens."""
    if model_family == "hermes":
        # Look for opening tag without closing tag, and incomplete JSON
        if "<tool_call>" in text and "</tool_call>" not in text:
            # Check if there's a JSON object that looks like it might be incomplete
            tool_start = text.find("<tool_call>")
            if tool_start != -1:
                content_after = text[tool_start + len("<tool_call>"):].strip()
                if content_after.startswith("{"):
                    # Count braces to see if it might be incomplete
                    open_braces = content_after.count("{")
                    close_braces = content_after.count("}")
                    if open_braces > close_braces:
                        return True
                    # Also check if it ends mid-quote or mid-value
                    if content_after.count('"') % 2 == 1:
                        return True
    elif model_family in ["mistral", "mistral_nemo"]:
        if "[TOOL_CALLS]" in text and not re.search(r'\[TOOL_CALLS\].*?\]', text):
            return True
    elif model_family == "llama3":
        if "<|python_tag|>" in text and ("<|eom_id|>" not in text and "<|eot_id|>" not in text):
            return True
    
    return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@lru_cache(maxsize=100)
def get_special_tokens(model_name: str):
    """Gets all special tokens for a Hugging Face model."""
    try:
        if model_name in models:
            _, tokenizer, _ = models[model_name]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        special_tokens = {"|begin_of_text|>"}
        return special_tokens
        
    except Exception as e:
        print(f"Error detecting special tokens for {model_name}: {str(e)}")
        return {"|begin_of_text|>"}


TOKENS_TO_REMOVE = ["|begin_of_text|>"]

@lru_cache(maxsize=1)
def get_cleanup_regex(special_tokens):
    """Create a compiled regex for all tokens to remove."""
    if not isinstance(special_tokens, frozenset):
        special_tokens = frozenset(special_tokens)
    
    all_tokens = set(special_tokens).union(TOKENS_TO_REMOVE)
    all_tokens = [re.escape(token) for token in all_tokens if token and token.strip()]
    
    if not all_tokens:
        return None
    
    all_tokens.sort(key=len, reverse=True)
    pattern = '|'.join(all_tokens)
    return re.compile(pattern)


def clean_special_tokens(text: str, special_tokens: set) -> str:
    """Removes specified special tokens from the generated text using regex."""
    if not special_tokens and not TOKENS_TO_REMOVE:
        return text
    
    regex = get_cleanup_regex(frozenset(special_tokens))
    
    if not regex:
        return text
    
    return regex.sub('', text)


def clean_role_markers(text: str) -> str:
    """Removes role markers and XML tags from the text."""
    role_markers = [
        "assistant:", "Assistant:", "ASSISTANT:", 
        "user:", "User:", "USER:", 
        "system:", "System:", "SYSTEM:"
    ]
    
    cleaned_text = text
    for marker in role_markers:
        if cleaned_text.lstrip().startswith(marker):
            cleaned_text = cleaned_text.lstrip().replace(marker, "", 1).lstrip()
    
    return cleaned_text


def get_stop_sequences_for_tools(model_name: str, tools: List[Tool], base_stop_sequences: List[str]) -> List[str]:
    """Get appropriate stop sequences that don't interfere with tool calls."""
    stop_sequences = base_stop_sequences.copy()
    
    model_family = detect_model_family(model_name)
    
    # CRITICAL FIX: Don't add tool call closing tags as stop sequences
    if model_family == "hermes":
        # Don't add </tool_call> as stop sequence - let model complete tool calls
        stop_sequences.extend(["<|im_end|>"])
    elif model_family == "llama3":
        stop_sequences.extend(["<|eot_id|>", "<|end_of_text|>"])
    elif model_family in ["mistral", "mistral_nemo"]:
        stop_sequences.extend(["[/INST]", "</s>"])
    
    # Add conversational stop sequences
    chat_stop_sequences = [
        "\nuser:", "\nassistant:", "\nsystem:",
        "<user>", "</user>", "<assistant>", "</assistant>", 
        "<s>", "</s>"
    ]
    stop_sequences.extend(chat_stop_sequences)
    
    return stop_sequences


@lru_cache(maxsize=100)
def get_stop_tokens(model_name: str):
    """Detects stop sequences for a Hugging Face model."""
    try:
        if model_name in models:
            _, tokenizer, _ = models[model_name]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        stop_tokens = set()

        if tokenizer.eos_token:
            stop_tokens.add(tokenizer.eos_token)

        if tokenizer.bos_token:
            stop_tokens.add(tokenizer.bos_token)

        if tokenizer.additional_special_tokens:
            stop_tokens.update(tokenizer.additional_special_tokens)

        tokenizer_json_path = tokenizer.init_kwargs.get("tokenizer_file")

        if tokenizer_json_path and os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)

            if "bos_token" in tokenizer_config and isinstance(tokenizer_config["bos_token"], dict):
                stop_tokens.add(tokenizer_config["bos_token"].get("content", ""))

            if "eos_token" in tokenizer_config and isinstance(tokenizer_config["eos_token"], dict):
                stop_tokens.add(tokenizer_config["eos_token"].get("content", ""))

            if "added_tokens_decoder" in tokenizer_config:
                for token_info in tokenizer_config["added_tokens_decoder"].values():
                    content = token_info.get("content")
                    if content and token_info.get("special", False):
                        if not content.startswith("<|reserved_special_token_"):
                            stop_tokens.add(content)

            if "chat_template" in tokenizer_config:
                chat_template = tokenizer_config["chat_template"]

                if "<|eot_id|>" in chat_template:
                    stop_tokens.add("<|eot_id|>")
                
                if "<|end_of_text|>" in chat_template:
                    stop_tokens.add("<|end_of_text|>")

                if "<|start_header_id|>" in chat_template and "<|end_header_id|>" in chat_template:
                    stop_tokens.add("<|end_header_id|>")

        return sorted(stop_tokens) if stop_tokens else None

    except Exception as e:
        traceback.print_exc()
        print(f"Error detecting stop tokens for {model_name}: {str(e)}")
        return None


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup complete")
    yield
    print("Application shutdown initiated, unloading models...")
    utils.unload_models(models)
    print("Models unloaded successfully")


app = FastAPI(lifespan=lifespan)
models = utils.load_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # DEBUG: Log incoming request
        #print("=" * 60)
        #print("🔍 DEBUG: /v1/chat/completions REQUEST:")
        #print(json.dumps({
        #    "model": request.model,
        #    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        #    "max_tokens": request.max_tokens,
        #    "temperature": request.temperature,
        #    "stream": request.stream,
        #    "tools": [tool.dict() for tool in request.tools] if request.tools else None,
        #    "tool_choice": request.tool_choice
        #}, indent=2))
        #print("=" * 60)
        
        model_name = request.model
        messages = request.messages
        max_new_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        top_k = request.top_k
        repetition_penalty = request.repetition_penalty
        stream = request.stream
        tools = request.tools or []
        tool_choice = request.tool_choice

        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        model_info = models[model_name]
        model, tokenizer, backend_config = model_info

        if not backend_config.public_api:
            raise HTTPException(status_code=403, detail=f"Access to model '{model_name}' is denied")

        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        conversation_history = format_conversation_with_tools(messages, tools, tokenizer, model_name)

        inputs = tokenizer(conversation_history, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        max_length = input_ids.shape[1] + max_new_tokens
        
        prompt_tokens = input_ids.shape[1]

        # FIXED: Use tool-aware stop sequences
        detected_stop_tokens = get_stop_tokens(model_name)
        base_stop_sequences = detected_stop_tokens or []
        
        if request.stop:
            if isinstance(request.stop, list):
                base_stop_sequences.extend(request.stop)
            else:
                base_stop_sequences.append(request.stop)
        
        # Get appropriate stop sequences that don't interfere with tool calls
        stop_sequences = get_stop_sequences_for_tools(model_name, tools, base_stop_sequences)

        if stream:
            #print("🔍 DEBUG: Using STREAMING response")
            return StreamingResponse(
                stream_generate_chat(
                    input_ids, attention_mask, model, tokenizer, 
                    stop_sequences, True, temperature, top_p, top_k, 
                    repetition_penalty, max_length, max_new_tokens, prompt_tokens,
                    tools, tool_choice
                ),
                media_type='text/event-stream'
            )
        else:
            #print("🔍 DEBUG: Using NON-STREAMING response")
            output_text, prompt_token_count, completion_token_count = await generate_text(
                input_ids, attention_mask, model, tokenizer, 
                stop_sequences, True, temperature, top_p, top_k, 
                repetition_penalty, max_length, max_new_tokens
            )            
            response = create_chat_completion_response(
                output_text, model_name, prompt_token_count, completion_token_count, tools, model_name
            )
            
            # DEBUG: Log outgoing response
            #print("=" * 60)
            #print("🔍 DEBUG: /v1/chat/completions RESPONSE:")
            #print(json.dumps(response, indent=2))
            #print("=" * 60)
            
            return JSONResponse(content=response)

    except Exception as e:
        error_response = {"error": str(e)}
        
        # DEBUG: Log error response
        #print("=" * 60)
        #print("🔍 DEBUG: /v1/chat/completions ERROR:")
        #print(json.dumps(error_response, indent=2))
        #print(f"🔍 DEBUG: Exception: {str(e)}")
        #print("=" * 60)
        
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content=error_response, status_code=500)


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    request_start_time = time.time()
    
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

        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        
        input_ids = inputs["input_ids"].to(config.DEVICE)
        attention_mask = inputs["attention_mask"].to(config.DEVICE)
        
        max_length = input_ids.shape[1] + max_new_tokens
        prompt_tokens = input_ids.shape[1]

        detected_stop_tokens = get_stop_tokens(model_name)
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
            
            total_time = time.time() - request_start_time
            print(f"Total request time: {total_time:.4f} seconds")
            
            response = create_completion_response(output_text, model_name, prompt_token_count, completion_token_count)
            return JSONResponse(content=response)

    except Exception as e:
        error_time = time.time() - request_start_time
        print(f"Error occurred after {error_time:.4f} seconds")
        
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


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


# =============================================================================
# RESPONSE CREATION FUNCTIONS
# =============================================================================

def create_chat_completion_response(text: str, model_name: str, prompt_tokens: int, completion_tokens: int, tools: List[Tool] = None, model_id: str = "") -> dict:
    """Create OpenAI-compatible chat completion response with proper tool call handling."""
    
    #print(f"🔍 DEBUG create_chat_completion_response:")
    #print(f"  Raw text: {repr(text[:200])}...")
    #print(f"  Has tools: {bool(tools)}")
    #print(f"  Model ID: {model_id}")
    
    # Step 1: Clean role markers from raw text
    cleaned_text = clean_role_markers(text)
    #print(f"  After role cleaning: {repr(cleaned_text[:200])}...")
    
    # Step 2: Try to extract tool call if tools are available
    tool_call = None
    final_content = cleaned_text
    
    if tools:
        tool_call = extract_tool_call_from_response(cleaned_text, tools, model_id)
        #print(f"  Extracted tool call: {tool_call}")
        
        if tool_call:
            # Step 3: Remove the raw tool call syntax from content
            model_family = detect_model_family(model_id)
            #print(f"  Model family: {model_family}")
            
            original_length = len(cleaned_text)
            
            if model_family == "hermes":
                # Remove complete <tool_call>...</tool_call> blocks
                final_content = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_text, flags=re.DOTALL)
                # Also remove incomplete tool calls (opening tag + any content until end or next tag)
                final_content = re.sub(r'<tool_call>\s*\{[^<]*?(?:\s*$|(?=<))', '', final_content, flags=re.DOTALL)
                # Remove any remaining opening tags
                final_content = re.sub(r'<tool_call>.*', '', final_content, flags=re.DOTALL)
                final_content = final_content.strip()
            elif model_family in ["mistral", "mistral_nemo"]:
                # Remove [TOOL_CALLS] blocks
                final_content = re.sub(r'\[TOOL_CALLS\].*?\]', '', cleaned_text, flags=re.DOTALL).strip()
            elif model_family == "llama3":
                # Remove various Llama 3 tool call formats
                final_content = re.sub(r'<\|python_tag\|>.*?<\|eom_id\|>', '', cleaned_text, flags=re.DOTALL).strip()
                final_content = re.sub(r'```json\s*\{.*?\}\s*```', '', final_content, flags=re.DOTALL).strip()
            else:
                # Remove generic JSON tool call format
                final_content = re.sub(r'\{"tool_call":\s*\{[^}]*\}\s*\}', '', cleaned_text).strip()
                final_content = re.sub(r'\{[^}]*"name"[^}]*"arguments"[^}]*\}', '', final_content).strip()
            
            #print(f"  After tool call cleaning: {repr(final_content[:200])}...")
            #print(f"  Text length: {original_length} -> {len(final_content)}")
    
    # Step 4: Build response following OpenAI standard
    choice = {
        "message": {
            "role": "assistant",
        },
        "index": 0,
        "finish_reason": "tool_calls" if tool_call else "stop"
    }
    
    if tool_call:
        # OpenAI standard: empty content when tool calls present
        choice["message"]["content"] = ""
        choice["message"]["tool_calls"] = [tool_call.dict()]
        print(f"  Final response has tool_calls, content is empty")
    else:
        # Regular text response
        choice["message"]["content"] = final_content
        #print(f"  Final response content: {repr(final_content[:100])}...")
    
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    #print(f"  Final response structure: {json.dumps(response, indent=2)}")
    return response


def create_completion_response(text: str, model_name: str, prompt_tokens: int, completion_tokens: int) -> dict:
    # Check if we should return chat-style format for no-code tools
    # You can add a config flag or detect based on user agent
    use_chat_format = True  # Set this based on your needs
    
    if use_chat_format:
        # Return chat completions format for better tool compatibility
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "chat.completion",  # Changed to chat.completion
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "message": {  # Changed from text to message
                        "role": "assistant",
                        "content": text
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
    else:
        # Original OpenAI completions format
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


# =============================================================================
# GENERATION FUNCTIONS (FIXED FOR INCOMPLETE TOOL CALLS)
# =============================================================================

async def stream_generate(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens, prompt_tokens) -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    buffer_size = config.STREAM_BUFFER_SIZE
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0
        delta_q = []
        token_buffer = []
        raw_buffer_text = ""
        cleaned_buffer_text = ""
        
        stop = False
        stopped_due_to_sequence = False
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
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)

            if "\ufffd" in token_text:
                delta_q += delta
                continue

            delta_q = []
            
            all_outputs += token_text
            generated_tokens += 1
            
            cleaned_token = clean_special_tokens(token_text, special_tokens)
            
            token_buffer.append((delta, token_text, cleaned_token))
            raw_buffer_text += token_text
            cleaned_buffer_text += cleaned_token
            
            
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_buffer_text:
                    stop_idx = cleaned_buffer_text.find(stop_seq)
                    
                    current_pos = 0
                    tokens_to_keep = []
                    for i, (d, raw, clean) in enumerate(token_buffer):
                        if current_pos + len(clean) > stop_idx:
                            if current_pos < stop_idx:
                                keep_chars = stop_idx - current_pos
                                if keep_chars > 0:
                                    partial_clean = clean[:keep_chars]
                                    if partial_clean:
                                        tokens_to_keep.append((d, raw, partial_clean))
                            break
                        else:
                            tokens_to_keep.append((d, raw, clean))
                            current_pos += len(clean)
                    
                    token_buffer = tokens_to_keep
                    stopped_due_to_sequence = True
                    
                    raw_buffer_text = "".join(t[1] for t in token_buffer)
                    cleaned_buffer_text = "".join(t[2] for t in token_buffer)
                    
                    stop_found = True
                    stop = True
                    break
            
            if len(token_buffer) >= buffer_size or stop_found or generated_tokens == max_new_tokens:
                while token_buffer and (len(token_buffer) > 1 or stop):
                    _, _, first_cleaned_token = token_buffer.pop(0)
                    
                    if first_cleaned_token:
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
                        
                        if usage:
                            response["usage"] = usage
                        
                        yield f"data: {json.dumps(response)}\n\n"
                        await asyncio.sleep(0)
                
                raw_buffer_text = "".join(t[1] for t in token_buffer)
                cleaned_buffer_text = "".join(t[2] for t in token_buffer)

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True
                
        if token_buffer and not stopped_due_to_sequence:
            _, _, last_cleaned_token = token_buffer[0]
            if last_cleaned_token:
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


async def stream_generate_chat(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens, prompt_tokens, tools: List[Tool] = None, tool_choice = "auto") -> Generator[str, None, None]:
    n_input_tokens = input_ids.shape[1]
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    buffer_size = config.STREAM_BUFFER_SIZE
    
    with model.inference_session(max_length=max_length) as session:
        all_outputs = ""
        generated_tokens = 0  
        delta_q = []
        
        token_buffer = []
        raw_buffer_text = ""
        cleaned_buffer_text = ""
        
        # NEW: Maintain full accumulated cleaned text (not just buffer)
        full_cleaned_text = ""
        
        stop = False
        stopped_due_to_sequence = False
        first_step = True
        start_time = int(time.time())
        initial_token = True
        accumulated_response = ""
        incomplete_tool_call_timeout = 0  # NEW: Track how long we wait for completion
        max_incomplete_wait = 5  # NEW: Max tokens to wait for completion

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
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )

            delta = outputs[0, n_input_tokens:].tolist()
            token_text = safe_decode(tokenizer, delta_q + delta)
            generated_tokens += 1

            if "\ufffd" in token_text:
                delta_q += delta
                continue

            delta_q = []
            
            all_outputs += token_text
            accumulated_response += token_text
            
            cleaned_token = clean_special_tokens(token_text, special_tokens)
            
            if initial_token:
                cleaned_token = clean_role_markers(cleaned_token)
                initial_token = False
                
            token_buffer.append((delta, token_text, cleaned_token))
            raw_buffer_text += token_text
            cleaned_buffer_text += cleaned_token
            
            # NEW: Also accumulate in full cleaned text
            full_cleaned_text += cleaned_token            
            
            # Enhanced tool call detection 
            tool_call_detected = False
            if tools:
                # Check for complete tool calls using FULL CLEANED text
                tool_call = extract_tool_call_from_response(full_cleaned_text, tools, tokenizer.name_or_path)
                if tool_call:
                    print(f"🔧 STREAMING: Complete tool call detected: {tool_call.function['name']}")
                    tool_call_detected = True
                    stop = True
            
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_buffer_text:
                    stop_idx = cleaned_buffer_text.find(stop_seq)
                    
                    current_pos = 0
                    tokens_to_keep = []
                    for i, (d, raw, clean) in enumerate(token_buffer):
                        if current_pos + len(clean) > stop_idx:
                            if current_pos < stop_idx:
                                keep_chars = stop_idx - current_pos
                                if keep_chars > 0:
                                    partial_clean = clean[:keep_chars]
                                    if partial_clean:
                                        tokens_to_keep.append((d, raw, partial_clean))
                            break
                        else:
                            tokens_to_keep.append((d, raw, clean))
                            current_pos += len(clean)
                    
                    token_buffer = tokens_to_keep
                    stopped_due_to_sequence = True
                    
                    raw_buffer_text = "".join(t[1] for t in token_buffer)
                    cleaned_buffer_text = "".join(t[2] for t in token_buffer)
                    
                    stop_found = True
                    stop = True
                    break
            
            if len(token_buffer) >= buffer_size or stop_found or generated_tokens == max_new_tokens or tool_call_detected:
                if tool_call_detected and tool_call:
                    # Clean the content using FULL CLEANED text for consistency
                    model_family = detect_model_family(tokenizer.name_or_path)
                    clean_content = full_cleaned_text
                    
                    if model_family == "hermes":
                        clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', clean_content, flags=re.DOTALL)
                        # Also remove incomplete tool calls (opening tag + any content until end or next tag)
                        clean_content = re.sub(r'<tool_call>\s*\{[^<]*?(?:\s*$|(?=<))', '', clean_content, flags=re.DOTALL)
                        # Remove any remaining opening tags
                        clean_content = re.sub(r'<tool_call>.*', '', clean_content, flags=re.DOTALL)
                        clean_content = clean_content.strip()
                    elif model_family in ["mistral", "mistral_nemo"]:
                        clean_content = re.sub(r'\[TOOL_CALLS\].*?\]', '', clean_content, flags=re.DOTALL).strip()
                    elif model_family == "llama3":
                        clean_content = re.sub(r'<\|python_tag\|>.*?<\|eom_id\|>', '', clean_content, flags=re.DOTALL).strip()
                        clean_content = re.sub(r'```json\s*\{.*?\}\s*```', '', clean_content, flags=re.DOTALL).strip()
                    else:
                        clean_content = re.sub(r'\{"tool_call":\s*\{[^}]*\}\s*\}', '', clean_content).strip()
                    
                    # Send any remaining content first (if any)
                    if clean_content.strip():
                        content_response = {
                            "id": f"chatcmpl-{start_time}",
                            "object": "chat.completion.chunk",
                            "created": start_time,
                            "model": model.__class__.__name__,
                            "choices": [{
                                "delta": {"content": clean_content},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(content_response)}\n\n"
                    
                    # Then send the tool call
                    tool_response = {
                        "id": f"chatcmpl-{start_time}",
                        "object": "chat.completion.chunk", 
                        "created": start_time,
                        "model": model.__class__.__name__,
                        "choices": [{
                            "delta": {"tool_calls": [tool_call.dict()]},
                            "index": 0,
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": generated_tokens,
                            "total_tokens": prompt_tokens + generated_tokens
                        }
                    }
                    
                    # DEBUG: Log streaming tool call response
                    #print("🔍 DEBUG: STREAMING TOOL CALL CHUNK:")
                    #print(json.dumps(tool_response, indent=2))
                    
                    yield f"data: {json.dumps(tool_response)}\n\n"
                    await asyncio.sleep(0)
                    break
                
                # Send regular content chunks (when not a tool call)
                if not tool_call_detected:
                    while token_buffer and (len(token_buffer) > 1 or stop):
                        _, _, first_cleaned_token = token_buffer.pop(0)
                        
                        if first_cleaned_token:
                            usage = None
                            if stop and not token_buffer:
                                usage = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": generated_tokens,
                                    "total_tokens": prompt_tokens + generated_tokens
                                }
                            
                            response = {
                                "id": f"chatcmpl-{start_time}",
                                "object": "chat.completion.chunk",
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
                            
                            if usage:
                                response["usage"] = usage
                            
                            # DEBUG: Log streaming content chunk (only first few for brevity)
                            #if generated_tokens <= 3:
                            #    print(f"🔍 DEBUG: STREAMING CONTENT CHUNK #{generated_tokens}:")
                            #    print(json.dumps(response, indent=2))
                            
                            yield f"data: {json.dumps(response)}\n\n"
                            await asyncio.sleep(0)
                
                raw_buffer_text = "".join(t[1] for t in token_buffer)
                cleaned_buffer_text = "".join(t[2] for t in token_buffer)

            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False

            if generated_tokens == max_new_tokens: 
                stop = True
                
        # NEW: After natural stopping, check for incomplete tool calls and try to complete them
        if not tool_call_detected and tools and (stopped_due_to_sequence or generated_tokens == max_new_tokens):
            model_family = detect_model_family(tokenizer.name_or_path)
            is_incomplete = is_tool_call_potentially_incomplete(full_cleaned_text, model_family)
            
            if is_incomplete:
                print(f"🔧 STREAMING: Generation stopped but incomplete tool call detected, attempting completion...")
                #print(f"🔍 DEBUG: Full cleaned text for incomplete check: {repr(full_cleaned_text[-200:])}")  # Debug last 200 chars
                # Try to extract and complete the incomplete tool call
                completed_tool_call = extract_tool_call_from_response(full_cleaned_text, tools, tokenizer.name_or_path)
                if completed_tool_call:
                    tool_call = completed_tool_call
                    tool_call_detected = True
                    print(f"🔧 STREAMING: Successfully completed tool call: {tool_call.function['name']}")
                else:
                    print(f"🔧 STREAMING: Could not complete incomplete tool call")
                
        # Handle final response based on what we detected
        if tool_call_detected and tool_call:
            # Send tool call response
            model_family = detect_model_family(tokenizer.name_or_path)
            clean_content = full_cleaned_text  # Use full text for cleaning
            
            if model_family == "hermes":
                clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', clean_content, flags=re.DOTALL)
                clean_content = re.sub(r'<tool_call>\s*\{[^<]*?(?:\s*$|(?=<))', '', clean_content, flags=re.DOTALL)
                clean_content = re.sub(r'<tool_call>.*', '', clean_content, flags=re.DOTALL)
                clean_content = clean_content.strip()
            elif model_family in ["mistral", "mistral_nemo"]:
                clean_content = re.sub(r'\[TOOL_CALLS\].*?\]', '', clean_content, flags=re.DOTALL).strip()
            elif model_family == "llama3":
                clean_content = re.sub(r'<\|python_tag\|>.*?<\|eom_id\|>', '', clean_content, flags=re.DOTALL).strip()
                clean_content = re.sub(r'```json\s*\{.*?\}\s*```', '', clean_content, flags=re.DOTALL).strip()
            else:
                clean_content = re.sub(r'\{"tool_call":\s*\{[^}]*\}\s*\}', '', clean_content).strip()
            
            # Send any remaining content first (if any)
            if clean_content.strip():
                content_response = {
                    "id": f"chatcmpl-{start_time}",
                    "object": "chat.completion.chunk",
                    "created": start_time,
                    "model": model.__class__.__name__,
                    "choices": [{
                        "delta": {"content": clean_content},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(content_response)}\n\n"
            
            # Send the tool call
            tool_response = {
                "id": f"chatcmpl-{start_time}",
                "object": "chat.completion.chunk", 
                "created": start_time,
                "model": model.__class__.__name__,
                "choices": [{
                    "delta": {"tool_calls": [tool_call.dict()]},
                    "index": 0,
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": generated_tokens,
                    "total_tokens": prompt_tokens + generated_tokens
                }
            }
            
            #print("🔍 DEBUG: STREAMING TOOL CALL CHUNK:")
            #print(json.dumps(tool_response, indent=2))
            
            yield f"data: {json.dumps(tool_response)}\n\n"
            await asyncio.sleep(0)
            
        elif token_buffer:
            # Send remaining content as regular text
            _, _, last_cleaned_token = token_buffer[0]
            if last_cleaned_token:
                response = {
                    "id": f"chatcmpl-{start_time}",
                    "object": "chat.completion.chunk",
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
        
        #print("Output: " + all_outputs)
        yield "data: [DONE]\n\n"


async def generate_text(input_ids, attention_mask, model, tokenizer, stop_sequences, do_sample, temperature, top_p, top_k, repetition_penalty, max_length, max_new_tokens) -> tuple:
    generate_start_time = time.time()
    
    prompt_tokens = input_ids.shape[1]
    n_input_tokens = input_ids.shape[1]
    all_outputs = ""
    cleaned_all_outputs = ""
    delta_q = []
    stop = False
    first_step = True
    generated_tokens = 0
    
    total_model_generate_time = 0
    total_decode_time = 0
    total_stop_check_time = 0
    
    special_tokens_start = time.time()
    special_tokens = get_special_tokens(tokenizer.name_or_path)
    special_tokens_time = time.time() - special_tokens_start
    
    session_start_time = time.time()
    
    with model.inference_session(max_length=max_length) as session:
        session_created_time = time.time()
        session_setup_time = session_created_time - session_start_time
        
        while not stop:
            token_gen_start_time = time.time()
            
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
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
            
            token_gen_time = time.time() - token_gen_start_time
            total_model_generate_time += token_gen_time
            
            if first_step:
                first_token_time = time.time() - generate_start_time
                print(f"Time to first token: {first_token_time:.4f} seconds")
            
            delta = outputs[0, n_input_tokens:].tolist()
            
            decode_start_time = time.time()
            token_text = safe_decode(tokenizer, delta_q + delta)
            decode_time = time.time() - decode_start_time
            total_decode_time += decode_time
            
            generated_tokens += 1
            
            stop_check_start = time.time()
            
            potential_text = all_outputs + token_text
            cleaned_potential = clean_special_tokens(potential_text, special_tokens)
            
            stop_found = False
            for stop_seq in stop_sequences:
                if stop_seq in cleaned_potential:
                    stop_idx = cleaned_potential.find(stop_seq)
                    all_outputs = potential_text[:len(potential_text) - len(token_text) + (stop_idx - len(cleaned_potential) + len(token_text))]
                    cleaned_all_outputs = cleaned_potential[:stop_idx]
                    print(f"Got stop seq: {stop_seq!r}, truncating output at token #{generated_tokens}")
                    stop_found = True
                    stop = True
                    break
            
            if not stop_found:
                if "\ufffd" in token_text:
                    delta_q += delta
                    continue
                
                delta_q = []
                all_outputs += token_text
                cleaned_all_outputs = cleaned_potential
            
            stop_check_time = time.time() - stop_check_start
            total_stop_check_time += stop_check_time
            
            if first_step:
                input_ids = None
                attention_mask = None
                n_input_tokens = 0
                first_step = False
            
            if generated_tokens == max_new_tokens: 
                stop = True
                print(f"Reached max tokens ({max_new_tokens})")                
    
    other_time = time.time() - generate_start_time - total_model_generate_time - total_decode_time - total_stop_check_time - special_tokens_time - session_setup_time
    
    print(f"Timing breakdown:")
    print(f"  Model generation: {total_model_generate_time:.4f}s ({total_model_generate_time/generated_tokens:.4f}s per token)")
    print(f"  Token decoding: {total_decode_time:.4f}s ({total_decode_time/generated_tokens:.4f}s per token)")
    print(f"  Stop sequence checking: {total_stop_check_time:.4f}s ({total_stop_check_time/generated_tokens:.4f}s per token)")
    print(f"  Special tokens lookup: {special_tokens_time:.4f}s")
    print(f"  Session setup: {session_setup_time:.4f}s")
    print(f"  Other operations: {other_time:.4f}s")
    print(f"Total generation time: {time.time() - generate_start_time:.4f} seconds")
    print(f"Tokens generated: {generated_tokens}")
    print(f"Generation speed: {generated_tokens / (time.time() - generate_start_time):.2f} tokens/second")
    
    return cleaned_all_outputs, prompt_tokens, generated_tokens


# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def handle_sigterm(signum, frame):
    print("Received SIGTERM. Initiating graceful shutdown...")
    utils.unload_models(models)
    print("Models unloaded successfully")
    sys.exit(0)


def handle_sigint(signum, frame):
    print("Received SIGINT. Initiating graceful shutdown...")
    utils.unload_models(models)
    print("Models unloaded successfully")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigint)