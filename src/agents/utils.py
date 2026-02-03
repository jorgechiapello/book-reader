import os
from crewai import LLM

def local_llm(model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434") -> LLM:
    """
    Create a local Ollama LLM connection via CrewAI.
    
    Args:
        model: Ollama model name (e.g., "llama3.2:3b", "llama3", "mistral")
        base_url: Ollama API endpoint
    
    Returns:
        Configured LLM instance
    """
    # Set environment variable to bypass OpenAI API key requirement
    os.environ["OPENAI_API_KEY"] = "NA"
    
    return LLM(
        model=f"ollama/{model}",
        base_url=base_url,
        api_key="NA"
    )

