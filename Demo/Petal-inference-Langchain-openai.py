from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Create an OpenAI instance with streaming enabled
llm = OpenAI(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    openai_api_base="http://localhost:8000/v1",   
    max_tokens=20,    
    temperature=0.9,
    **{"streaming": True}  # Enable streaming
).bind(logprobs=False)

# Use the LLM stream method for text completion
stream = llm.stream("Once upon a time, a brave knight")

# Retrieve and print the streaming output in real-time
print("LLM result: Once upon a time, a brave knight ")
for chunk in stream:
    print(chunk, end="", flush=True)  # Print each chunk of text as it streams

# Create a ChatOpenAI instance with streaming enabled
chat_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    openai_api_base="http://localhost:8000/v1",   
    max_tokens=50,      
    temperature=0.1,  
    **{"streaming": True}  # Enable streaming
).bind(logprobs=False)


# Stream the response from the ChatOpenAI model
print("\n\nuser: What is the capital of france?")

# Use the stream method to receive real-time chat completion
stream = chat_llm.stream(
    "system: You are a geography professor. Answer the user question very briefly.\nuser: What is the capital of France?\nassistant:\n",
    **{"stop": ["<|eot_id|>"]}
)

for chunk in stream:
    # Each chunk is an object that contains 'content' as an attribute
    if hasattr(chunk, 'content'):
        content = chunk.content
        if "<|eot_id|>" in content:
            # Stop printing if <|eot_id|> tag is found
            content = content.split("<|eot_id|>")[0]
            print(content, end="", flush=True)
            break
        else:
            print(content, end="", flush=True)