import os
import base64
from typing import Any, Optional, List
from langchain_openai import ChatOpenAI

class ModelFactory:
    """
    Factory class to create instances of LLMs based on the provider.
    """
    
    # Vision-capable models
    VISION_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"],
        "openrouter": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini", 
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "google/gemini-pro-vision",
            "google/gemini-1.5-pro",
            "google/gemini-1.5-flash",
        ],
        "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"],
    }
    
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
    def get_vision_model(provider: str = "openrouter", model_name: Optional[str] = None, temperature: float = 0.3) -> Any:
        """
        Get a vision-capable model for image analysis.
        
        Args:
            provider: The model provider
            model_name: Specific model name, or None to auto-select
            temperature: Temperature for generation
        
        Returns:
            A vision-capable chat model
        """
        provider = provider.lower()
        
        # Auto-select a vision model if not specified
        if model_name is None:
            vision_models = ModelFactory.VISION_MODELS.get(provider, [])
            if not vision_models:
                raise ValueError(f"No vision models available for provider: {provider}")
            model_name = vision_models[0]  # Use first available
        
        return ModelFactory.get_model(provider, model_name, temperature)
    
    @staticmethod
    def is_vision_capable(provider: str, model_name: str) -> bool:
        """Check if a model supports vision/image inputs."""
        provider = provider.lower()
        vision_models = ModelFactory.VISION_MODELS.get(provider, [])
        return model_name in vision_models or any(v in model_name for v in vision_models)

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
