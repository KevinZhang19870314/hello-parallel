"""
OpenAI Chat service implementation
"""
import os
from typing import List, Optional, Dict, Any

# 尝试导入openai，但如果没有安装则不会失败
try:
    import openai
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = object  # 类型占位符
    ChatCompletion = Dict[str, Any]  # 类型占位符

from ai_service.base import ChatService
from ai_service.registry import register_service


class OpenAIChatService(ChatService):
    """
    Chat service implementation using OpenAI's API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        provider: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        **config
    ):
        """
        Initialize the OpenAI chat service
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (defaults to OPENAI_ORG env var)
            provider: Provider name (for registry identification)
            client: Pre-created AsyncOpenAI client instance (if provided, api_key and organization are ignored)
            **config: Additional configuration options
        """
        super().__init__(provider=provider, **config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The openai package is required to use OpenAIChatService. "
                "Please install it with `pip install openai`."
            )
        
        # Use pre-created client if provided
        if client is not None:
            self.client = client
        else:
            # Set up API credentials
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.organization = organization or os.getenv("OPENAI_ORG")
            
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided either as an argument or through the OPENAI_API_KEY environment variable")
            
            # Initialize the async client
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization if self.organization else None,
                **config
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **options
    ) -> str:
        """
        Generate a response using OpenAI's chat completion API
        
        Args:
            prompt: The user message
            system_prompt: System message to set behavior
            model: The model to use (None for default)
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            **options: Additional OpenAI-specific options
            
        Returns:
            The generated response text
        """
        try:
            # Use default model if none specified
            if model is None:
                model = self.default_model
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare the API call parameters
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **options
            }
            
            # Add max_tokens if specified
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Make the API call
            response: ChatCompletion = await self.client.chat.completions.create(**params)
            
            # Extract and return the response content
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def available_models(self) -> List[str]:
        """Get available models"""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
    
    @property
    def default_model(self) -> str:
        """Get default model"""
        return "gpt-3.5-turbo"


# Register the service with the registry if OpenAI is available
if OPENAI_AVAILABLE:
    register_service("chat", "openai", OpenAIChatService) 