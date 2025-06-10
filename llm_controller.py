# llm_controller.py
"""
LLMController - Universal LangChain Model Switcher
A unified interface for seamlessly switching between different LLM providers
while maintaining full LangChain compatibility.

Usage:
    from llm_controller import LLMController
    
    # Initialize with any provider
    llm = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
    response = llm.invoke("Hello!")
    
    # Switch providers seamlessly
    llm.switch_model(llm="gpt-4", provider="openai")
    response = llm.invoke("Same interface, different model!")
"""

import os
from typing import Dict, Any, Optional, Union, List, Iterator
import requests
import json

# Import LangChain components with fallbacks for different versions
try:
    # Try new LangChain structure first (0.1+)
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable
    LANGCHAIN_NEW_STRUCTURE = True
except ImportError:
    # Fallback to legacy structure
    try:
        from langchain.llms import Ollama
        from langchain.chat_models import ChatOpenAI, ChatAnthropic
        from langchain.schema import BaseMessage, AIMessage, HumanMessage
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from langchain.llms.base import LLM
        from langchain.chat_models.base import BaseChatModel
        try:
            from langchain_core.runnables import Runnable
        except ImportError:
            # Create a minimal Runnable base class for older versions
            class Runnable:
                def __init__(self, **kwargs):
                    pass
        BaseLanguageModel = BaseChatModel
        LANGCHAIN_NEW_STRUCTURE = False
    except ImportError as e:
        raise ImportError(
            "LangChain is required. Install with: "
            "pip install langchain-core langchain-openai langchain-anthropic"
        ) from e


class SimpleChain:
    """
    Simple chain implementation for fallback when RunnableSequence fails
    Provides basic pipeline functionality for older LangChain versions
    """
    
    def __init__(self, first, last):
        self.first = first
        self.last = last
    
    def invoke(self, input, config=None, **kwargs):
        """Run the chain: first component then second component"""
        try:
            intermediate = self.first.invoke(input, config, **kwargs)
            return self.last.invoke(intermediate, config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Chain execution failed: {e}") from e
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Async version of invoke"""
        try:
            intermediate = await self.first.ainvoke(input, config, **kwargs)
            return await self.last.ainvoke(intermediate, config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Async chain execution failed: {e}") from e
    
    def __or__(self, other):
        """Support chaining multiple components"""
        return SimpleChain(self, other)
    
    def __repr__(self):
        return f"SimpleChain({self.first} | {self.last})"


class LLMController(Runnable):
    """
    A unified LLM controller that provides seamless switching between providers
    while maintaining full LangChain compatibility for invoke(), LangGraph, etc.
    
    Supports: OpenAI, Anthropic (Claude), Grok (X.AI), Ollama, Hugging Face
    
    Example:
        controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
        response = controller.invoke("Hello world!")
        
        # Switch providers
        controller.switch_model(llm="gpt-4", provider="openai")
        response = controller.invoke("Same interface!")
    """
    
    def __init__(self, llm: str = "gpt-3.5-turbo", provider: str = "openai", **kwargs):
        """
        Initialize the LLM Controller
        
        Args:
            llm: Model name (e.g., "claude-3-sonnet-20240229", "gpt-4")
            provider: Provider name ("claude", "openai", "ollama", "grok", "huggingface")
            **kwargs: Additional parameters passed to underlying models
        """
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        
        self.llm_name = llm
        self.provider = provider
        self._model_kwargs = kwargs
        
        # Provider factory mapping
        self.model_configs = {
            "openai": self._create_openai_model,
            "claude": self._create_claude_model,
            "anthropic": self._create_claude_model,  # Alias for claude
            "grok": self._create_grok_model,
            "xai": self._create_grok_model,  # Alias for grok
            "ollama": self._create_ollama_model,
            "huggingface": self._create_huggingface_model,
            "hf": self._create_huggingface_model,  # Alias for huggingface
        }
        
        self._current_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on provider and llm name"""
        if self.provider not in self.model_configs:
            available_providers = ", ".join(self.model_configs.keys())
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                f"Available providers: {available_providers}"
            )
        
        try:
            self._current_model = self.model_configs[self.provider](self.llm_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def _create_openai_model(self, model_name: str) -> BaseLanguageModel:
        """Create OpenAI model"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "api_key": api_key,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Only add max_tokens if it's specified and not None
        max_tokens = self._model_kwargs.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add other parameters, excluding the ones we've already handled
        for k, v in self._model_kwargs.items():
            if k not in ["temperature", "max_tokens"] and v is not None:
                params[k] = v
        
        return ChatOpenAI(**params)
    
    def _create_claude_model(self, model_name: str) -> BaseLanguageModel:
        """Create Anthropic Claude model"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "api_key": api_key,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Only add max_tokens if it's specified and not None
        max_tokens = self._model_kwargs.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add other parameters, excluding the ones we've already handled
        for k, v in self._model_kwargs.items():
            if k not in ["temperature", "max_tokens"] and v is not None:
                params[k] = v
        
        return ChatAnthropic(**params)
    
    def _create_ollama_model(self, model_name: str) -> BaseLanguageModel:
        """Create Ollama model (local)"""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "base_url": base_url,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Add other parameters, excluding temperature and None values
        for k, v in self._model_kwargs.items():
            if k != "temperature" and v is not None:
                params[k] = v
        
        return Ollama(**params)
    
    def _create_grok_model(self, model_name: str) -> BaseLanguageModel:
        """Create Grok (X.AI) model"""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        
        return GrokChatModel(
            model_name=model_name,
            api_key=api_key,
            **self._model_kwargs
        )
    
    def _create_huggingface_model(self, model_name: str) -> BaseLanguageModel:
        """Create Hugging Face model"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        
        return HuggingFaceChatModel(
            model_name=model_name,
            api_key=api_key,
            **self._model_kwargs
        )
    
    def switch_model(self, llm: str, provider: str = None, **kwargs):
        """
        Switch to a different model/provider
        
        Args:
            llm: New model name
            provider: New provider (optional, keeps current if not specified)
            **kwargs: Additional model parameters to update
        """
        if provider:
            self.provider = provider
        self.llm_name = llm
        
        # Update model kwargs if provided
        if kwargs:
            self._model_kwargs.update(kwargs)
        
        self._initialize_model()
    
    # Core Runnable interface methods
    def invoke(self, input, config=None, **kwargs):
        """
        Invoke the current model
        
        Args:
            input: Input prompt or messages
            config: Optional configuration
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        try:
            return self._current_model.invoke(input, config, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error invoking {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Async version of invoke"""
        try:
            if hasattr(self._current_model, 'ainvoke'):
                return await self._current_model.ainvoke(input, config, **kwargs)
            else:
                # Fallback for models that don't support async
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self.invoke(input, config, **kwargs)
                )
        except Exception as e:
            raise RuntimeError(
                f"Error in async invoke for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def stream(self, input, config=None, **kwargs) -> Iterator[Any]:
        """
        Stream responses from the current model
        
        Args:
            input: Input prompt or messages
            config: Optional configuration
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            return self._current_model.stream(input, config, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Error streaming from {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def astream(self, input, config=None, **kwargs):
        """Async version of stream"""
        try:
            if hasattr(self._current_model, 'astream'):
                async for chunk in self._current_model.astream(input, config, **kwargs):
                    yield chunk
            else:
                # Fallback: convert sync stream to async
                for chunk in self.stream(input, config, **kwargs):
                    yield chunk
        except Exception as e:
            raise RuntimeError(
                f"Error in async stream for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def batch(self, inputs, config=None, **kwargs):
        """
        Process multiple inputs in batch
        
        Args:
            inputs: List of inputs
            config: Optional configuration
            **kwargs: Additional parameters
            
        Returns:
            List of responses
        """
        try:
            if hasattr(self._current_model, 'batch'):
                return self._current_model.batch(inputs, config, **kwargs)
            else:
                # Fallback: process one by one
                return [self.invoke(input, config, **kwargs) for input in inputs]
        except Exception as e:
            raise RuntimeError(
                f"Error in batch processing for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def abatch(self, inputs, config=None, **kwargs):
        """Async version of batch"""
        try:
            if hasattr(self._current_model, 'abatch'):
                return await self._current_model.abatch(inputs, config, **kwargs)
            else:
                # Fallback: process all async
                import asyncio
                tasks = [self.ainvoke(input, config, **kwargs) for input in inputs]
                return await asyncio.gather(*tasks)
        except Exception as e:
            raise RuntimeError(
                f"Error in async batch for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    # Pipeline operator support for LangChain chains
    def __or__(self, other):
        """Support for | operator: controller | parser"""
        try:
            # Try to use LangChain's RunnableSequence if available
            if LANGCHAIN_NEW_STRUCTURE:
                from langchain_core.runnables import RunnableSequence
                return RunnableSequence(first=self, last=other)
            else:
                return SimpleChain(self, other)
        except (ImportError, Exception):
            # Fall back to SimpleChain if RunnableSequence fails
            return SimpleChain(self, other)
    
    def __ror__(self, other):
        """Support for | operator: prompt | controller"""
        try:
            # Try to use LangChain's RunnableSequence if available
            if LANGCHAIN_NEW_STRUCTURE:
                from langchain_core.runnables import RunnableSequence
                return RunnableSequence(first=other, last=self)
            else:
                return SimpleChain(other, self)
        except (ImportError, Exception):
            # Fall back to SimpleChain if RunnableSequence fails
            return SimpleChain(other, self)
    
    # Method delegation for backwards compatibility
    def __getattr__(self, name):
        """
        Delegate any missing methods to the current model
        This ensures compatibility with all LangChain features
        """
        if self._current_model and hasattr(self._current_model, name):
            attr = getattr(self._current_model, name)
            # If it's a method, wrap it to maintain error context
            if callable(attr):
                def wrapper(*args, **kwargs):
                    try:
                        return attr(*args, **kwargs)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error calling {name} on {self.provider} model: {e}"
                        ) from e
                return wrapper
            return attr
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
    @property
    def current_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            "provider": self.provider,
            "model": self.llm_name,
            "type": type(self._current_model).__name__,
            "langchain_structure": "new" if LANGCHAIN_NEW_STRUCTURE else "legacy"
        }
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for LangChain compatibility"""
        return f"llm_controller_{self.provider}"
    
    def __repr__(self):
        return f"LLMController(provider='{self.provider}', model='{self.llm_name}')"
    
    def __str__(self):
        return f"LLMController[{self.provider}:{self.llm_name}]"


class GrokChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Grok (X.AI) API"""
    
    def __init__(self, model_name: str = "grok-beta", api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("XAI_API_KEY is required for Grok models")
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult
        
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
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Grok API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Grok API response format: {e}") from e
    
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
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required for Hugging Face models")
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult
        
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
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            else:
                content = str(result)
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Hugging Face API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Hugging Face API response format: {e}") from e
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"


# Convenience functions for quick setup
def create_controller(provider: str, model: str = None, **kwargs) -> LLMController:
    """
    Convenience function to create a controller with sensible defaults
    
    Args:
        provider: Provider name ("claude", "openai", "ollama", etc.)
        model: Model name (uses provider default if not specified)
        **kwargs: Additional parameters
    
    Returns:
        Configured LLMController
    """
    # Provider defaults
    defaults = {
        "claude": "claude-3-sonnet-20240229",
        "openai": "gpt-3.5-turbo",
        "ollama": "llama2",
        "grok": "grok-beta",
        "huggingface": "microsoft/DialoGPT-medium"
    }
    
    if not model:
        model = defaults.get(provider)
        if not model:
            raise ValueError(f"No default model for provider '{provider}'. Please specify a model.")
    
    return LLMController(llm=model, provider=provider, **kwargs)


# Example usage
if __name__ == "__main__":
    print("LLMController - Universal LangChain Model Switcher")
    print("=" * 50)
    
    try:
        # Example with environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create controller
        controller = create_controller("claude", temperature=0.8)
        print(f"Created: {controller}")
        print(f"Model info: {controller.current_model_info}")
        
        # Test basic functionality
        response = controller.invoke("Say hello in one word.")
        print(f"Response: {response.content}")
        
        # Switch models
        controller.switch_model("gpt-3.5-turbo", "openai")
        print(f"Switched to: {controller}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure you have API keys set in your environment variables.")