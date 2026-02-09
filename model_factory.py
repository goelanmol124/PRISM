import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI

class ModelFactory:
    """
    Factory class to create instances of LLMs based on the provider.
    """
    
    @staticmethod
    def get_model(provider: str, model_name: str, temperature: float = 0.7, **kwargs: Any) -> Any:
        """
        Factory method to get the specified LLM model.
        
        Args:
            provider: The model provider ('openai', 'openrouter', 'gemini').
            model_name: The name of the model.
            temperature: The temperature for the model.
            **kwargs: Additional arguments to pass to the model constructor.
        
        Returns:
            An instance of a LangChain ChatModel.
        """
        provider = provider.lower()
        
        if provider == "openai":
            return ModelFactory._get_openai_model(model_name, temperature, **kwargs)
        elif provider == "openrouter":
            return ModelFactory._get_openrouter_model(model_name, temperature, **kwargs)
        elif provider == "gemini":
            return ModelFactory._get_gemini_model(model_name, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _get_openai_model(model_name: str, temperature: float, **kwargs) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
            **kwargs
        )

    @staticmethod
    def _get_openrouter_model(model_name: str, temperature: float, **kwargs) -> ChatOpenAI:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
            
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            **kwargs
        )

    @staticmethod
    def _get_gemini_model(model_name: str, temperature: float, **kwargs) -> Any:
        # Check if langchain-google-genai is installed
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
                
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai package is not installed. "
                "Please install it using `pip install langchain-google-genai` to use Gemini models directly."
            )
