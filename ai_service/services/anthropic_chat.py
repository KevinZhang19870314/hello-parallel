"""
Anthropic (Claude) Chat service implementation
"""
import os
from typing import List, Optional, Dict, Any

from ai_service.base import ChatService
from ai_service.registry import register_service

# Try to import anthropic, but don't fail if it's not installed
try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = object  # type placeholder
    Message = Dict[str, Any]  # type placeholder


class AnthropicChatService(ChatService):
    """
    Chat service implementation using Anthropic's Claude API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
        **config
    ):
        """
        Initialize the Anthropic chat service
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            provider: Provider name (for registry identification)
            client: Pre-created AsyncAnthropic client instance (if provided, api_key is ignored)
            **config: Additional configuration options
        """
        super().__init__(provider=provider, **config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The anthropic package is required to use AnthropicChatService. "
                "Please install it with `pip install anthropic`."
            )
        
        # Use pre-created client if provided
        if client is not None:
            self.client = client
        else:
            # Set up API credentials
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                raise ValueError("Anthropic API key must be provided either as an argument or through the ANTHROPIC_API_KEY environment variable")
            
            # Initialize the async client
            self.client = AsyncAnthropic(api_key=self.api_key, **config)
    
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
        Generate a response using Anthropic's Claude API
        
        Args:
            prompt: The user message
            system_prompt: System message to set behavior
            model: The model to use (None for default)
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            **options: Additional Anthropic-specific options
            
        Returns:
            The generated response text
        """
        try:
            # Use default model if none specified
            if model is None:
                model = self.default_model
            
            # Prepare the API call parameters
            params: Dict[str, Any] = {
                "model": model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                **options
            }
            
            # Add max_tokens if specified
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Make the API call
            response: Message = await self.client.messages.create(**params)
            
            # Extract and return the response content
            return response.content[0].text if response.content else ""
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def available_models(self) -> List[str]:
        """Get available models"""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
    
    @property
    def default_model(self) -> str:
        """Get default model"""
        return "claude-3-haiku-20240307"


# Register the service with the registry if Anthropic is available
if ANTHROPIC_AVAILABLE:
    register_service("chat", "anthropic", AnthropicChatService) 