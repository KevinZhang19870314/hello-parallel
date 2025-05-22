"""
OpenAI Embedding service implementation
"""
import os
from typing import List, Optional, Dict, Any, Union

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = object  # type placeholder

from ai_service.base import EmbeddingService
from ai_service.registry import register_service


class OpenAIEmbeddingService(EmbeddingService):
    """
    Embedding service implementation using OpenAI's API
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
        Initialize the OpenAI embedding service
        
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
                "The openai package is required to use OpenAIEmbeddingService. "
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
    
    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **options
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the input text
        
        Args:
            text: The text or list of texts to embed
            model: The model to use (None for default)
            **options: Additional OpenAI-specific options
            
        Returns:
            The embedding vector(s)
        """
        try:
            # Use default model if none specified
            if model is None:
                model = self.default_model
            
            # Prepare the input - ensure it's a list
            input_texts = [text] if isinstance(text, str) else text
            
            # Prepare the API call parameters
            params: Dict[str, Any] = {
                "model": model,
                "input": input_texts,
                **options
            }
            
            # Make the API call
            response = await self.client.embeddings.create(**params)
            
            # Extract the embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Return a single embedding if input was a string, otherwise return the list
            if isinstance(text, str):
                return embeddings[0]
            else:
                return embeddings
        except Exception as e:
            if isinstance(text, str):
                return [0.0]  # Return a dummy embedding on error for a single text
            else:
                return [[0.0] for _ in text]  # Return dummy embeddings for multiple texts
    
    @property
    def available_models(self) -> List[str]:
        """Get available models"""
        return [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
    
    @property
    def default_model(self) -> str:
        """Get default model"""
        return "text-embedding-3-small"


# Register the service with the registry if OpenAI is available
if OPENAI_AVAILABLE:
    register_service("embedding", "openai", OpenAIEmbeddingService) 