import os
from typing import Dict, Any, Optional, Union, List
from langchain.llms import OpenAI, Ollama
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import BaseLanguageModel, BaseMessage, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
import requests
import json

class LLMController(BaseChatModel):
    """
    A unified LLM controller that provides seamless switching between providers
    while maintaining full LangChain compatibility for invoke(), LangGraph, etc.
    """
    
    def __init__(self, llm: str = "gpt-3.5-turbo", provider: str = "openai", **kwargs):
        super().__init__(**kwargs)
        self.llm_name = llm
        self.provider = provider
        self.model_configs = {
            "openai": self._create_openai_model,
            "claude": self._create_claude_model,
            "anthropic": self._create_claude_model,  # alias
            "grok": self._create_grok_model,
            "xai": self._create_grok_model,  # alias
            "ollama": self._create_ollama_model,
            "huggingface": self._create_huggingface_model,
            "hf": self._create_huggingface_model,  # alias
        }
        self._current_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on provider and llm name"""
        if self.provider not in self.model_configs:
            raise ValueError(f"Unsupported provider: {self.provider}. "
                           f"Supported: {list(self.model_configs.keys())}")
        
        self._current_model = self.model_configs[self.provider](self.llm_name)
    
    def _create_openai_model(self, model_name: str) -> BaseLanguageModel:
        """Create OpenAI model"""
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
    
    def _create_claude_model(self, model_name: str) -> BaseLanguageModel:
        """Create Anthropic Claude model"""
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7
        )
    
    def _create_ollama_model(self, model_name: str) -> BaseLanguageModel:
        """Create Ollama model"""
        return Ollama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    
    def _create_grok_model(self, model_name: str) -> BaseLanguageModel:
        """Create Grok model"""
        return GrokChatModel(
            model_name=model_name,
            api_key=os.getenv("XAI_API_KEY")
        )
    
    def _create_huggingface_model(self, model_name: str) -> BaseLanguageModel:
        """Create Hugging Face model"""
        return HuggingFaceChatModel(
            model_name=model_name,
            api_key=os.getenv("HUGGINGFACE_API_KEY")
        )
    
    def switch_model(self, llm: str, provider: str = None):
        """Switch to a different model/provider"""
        if provider:
            self.provider = provider
        self.llm_name = llm
        self._initialize_model()
    
    # LangChain BaseChatModel interface methods
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        """Generate method required by BaseChatModel"""
        return self._current_model._generate(messages, stop, run_manager, **kwargs)
    
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return f"llm_controller_{self.provider}"
    
    # Delegate all other methods to the current model
    def __getattr__(self, name):
        """Delegate any missing methods to the current model"""
        if self._current_model and hasattr(self._current_model, name):
            return getattr(self._current_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke method for LangChain compatibility"""
        return self._current_model.invoke(input, config, **kwargs)
    
    def ainvoke(self, input, config=None, **kwargs):
        """Async invoke method"""
        return self._current_model.ainvoke(input, config, **kwargs)
    
    def stream(self, input, config=None, **kwargs):
        """Stream method for streaming responses"""
        return self._current_model.stream(input, config, **kwargs)
    
    def astream(self, input, config=None, **kwargs):
        """Async stream method"""
        return self._current_model.astream(input, config, **kwargs)
    
    def batch(self, inputs, config=None, **kwargs):
        """Batch method for processing multiple inputs"""
        return self._current_model.batch(inputs, config, **kwargs)
    
    def abatch(self, inputs, config=None, **kwargs):
        """Async batch method"""
        return self._current_model.abatch(inputs, config, **kwargs)
    
    @property
    def current_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            "provider": self.provider,
            "model": self.llm_name,
            "type": type(self._current_model).__name__
        }


class GrokChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Grok (X.AI) API"""
    
    def __init__(self, model_name: str = "grok-beta", api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain.schema import ChatGeneration, ChatResult
        
        # Convert LangChain messages to API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
                api_messages.append({"role": "user", "content": str(msg.content)})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": api_messages,
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"Grok API error: {response.status_code} - {response.text}")
    
    @property
    def _llm_type(self) -> str:
        return "grok_chat"


class HuggingFaceChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Hugging Face Inference API"""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain.schema import ChatGeneration, ChatResult
        
        # Combine messages into a single prompt for non-chat models
        prompt = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages])
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False
            }
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            else:
                content = str(result)
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"


# Example usage functions
def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage ===")
    
    # Create controller with OpenAI
    llm = LLMController(llm="gpt-3.5-turbo", provider="openai")
    
    # Use like any LangChain model
    response = llm.invoke("What is the capital of France?")
    print(f"OpenAI Response: {response.content}")
    
    # Switch to Claude
    llm.switch_model(llm="claude-3-sonnet-20240229", provider="claude")
    response = llm.invoke("What is the capital of France?")
    print(f"Claude Response: {response.content}")


def example_langchain_integration():
    """Example with LangChain chains and prompts"""
    print("\n=== LangChain Integration ===")
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts simply."),
        ("human", "Explain {topic} in simple terms.")
    ])
    
    # Create controller
    llm = LLMController(llm="gpt-3.5-turbo", provider="openai")
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Run chain
    result = chain.invoke({"topic": "quantum computing"})
    print(f"Chain result: {result}")


def example_langgraph_compatibility():
    """Example showing LangGraph compatibility"""
    print("\n=== LangGraph Compatibility ===")
    
    # The LLMController can be used anywhere a LangChain model is expected
    llm = LLMController(llm="gpt-4", provider="openai")
    
    # Example of using in a simple agent-like pattern
    from langchain.schema import HumanMessage, AIMessage
    
    messages = [
        HumanMessage(content="Hello! Can you help me with Python?"),
    ]
    
    response = llm.invoke(messages)
    print(f"LangGraph-style usage: {response.content}")
    
    # Switch provider mid-conversation
    llm.switch_model(llm="claude-3-sonnet-20240229", provider="claude")
    
    messages.append(response)
    messages.append(HumanMessage(content="Now explain list comprehensions."))
    
    response2 = llm.invoke(messages)
    print(f"After switching to Claude: {response2.content}")


def example_multiple_providers():
    """Example testing multiple providers"""
    print("\n=== Multiple Provider Test ===")
    
    test_prompt = "Write a haiku about programming."
    
    providers_models = [
        ("openai", "gpt-3.5-turbo"),
        ("claude", "claude-3-sonnet-20240229"),
        ("ollama", "llama2"),
        # ("grok", "grok-beta"),  # Uncomment if you have XAI API key
        # ("huggingface", "microsoft/DialoGPT-medium"),  # Uncomment if you have HF API key
    ]
    
    for provider, model in providers_models:
        try:
            llm = LLMController(llm=model, provider=provider)
            response = llm.invoke(test_prompt)
            print(f"\n{provider.upper()} ({model}):")
            print(f"Response: {response.content}")
            print(f"Model info: {llm.current_model_info}")
        except Exception as e:
            print(f"\n{provider.upper()} failed: {e}")


def main():
    """Main function demonstrating all features"""
    print("LLMController - Unified LangChain Interface")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_langchain_integration()
        example_langgraph_compatibility()
        example_multiple_providers()
        
    except Exception as e:
        print(f"Error in examples: {e}")
        print("Make sure you have the required API keys set as environment variables:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- XAI_API_KEY (for Grok)")
        print("- HUGGINGFACE_API_KEY")


if __name__ == "__main__":
    main()