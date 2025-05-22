"""
Base classes and interfaces for AI services
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar

# Type variables for better type hints
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')

class AIService(ABC):
    """
    Abstract base class for all AI services
    
    This interface defines the basic structure for any AI service,
    regardless of the underlying provider or service type.
    """
    
    @abstractmethod
    async def process(self, input_data: T_Input, **options) -> T_Output:
        """
        Process the input and return the result
        
        Args:
            input_data: The input data to process
            **options: Additional options for processing
            
        Returns:
            The processed result
        """
        pass
    
    @property
    @abstractmethod
    def service_type(self) -> str:
        """
        Get the type of service (e.g., 'chat', 'embedding', 'agent')
        
        Returns:
            The service type as a string
        """
        pass
    
    @property
    def service_name(self) -> str:
        """
        Get the name of the service implementation
        
        Returns:
            The service name as a string
        """
        return self.__class__.__name__
    
    @property
    def provider_name(self) -> Optional[str]:
        """
        Get the name of the underlying provider, if applicable
        
        Returns:
            The provider name or None if not applicable
        """
        return getattr(self, "provider", None)
    
    def __str__(self) -> str:
        """String representation of the service"""
        provider = f" ({self.provider_name})" if self.provider_name else ""
        return f"{self.service_name}{provider} - {self.service_type} service"


class ChatService(AIService):
    """
    Base class for chat-based AI services
    
    Provides a standardized interface for chat completion services
    from various providers like OpenAI, Anthropic, etc.
    """
    
    def __init__(self, provider: Optional[str] = None, **config):
        """
        Initialize the chat service
        
        Args:
            provider: Name of the underlying provider (if applicable)
            **config: Additional configuration options
        """
        self.provider = provider
        self.config = config
    
    @abstractmethod
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
        Generate a chat completion response
        
        Args:
            prompt: The user message to send
            system_prompt: System message to set the behavior of the assistant
            model: The model to use (None for default)
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            **options: Additional provider-specific options
            
        Returns:
            The generated response as a string
        """
        pass
    
    async def process(self, input_data: Union[str, Dict[str, Any]], **options) -> str:
        """
        Process the input message and return the chat response
        
        Args:
            input_data: Either a string prompt or a dictionary with prompt and options
            **options: Additional options for processing
            
        Returns:
            The chat response
        """
        if isinstance(input_data, str):
            return await self.generate(prompt=input_data, **options)
        elif isinstance(input_data, dict):
            prompt = input_data.pop("prompt")
            merged_options = {**input_data, **options}
            return await self.generate(prompt=prompt, **merged_options)
        else:
            raise TypeError(f"Expected str or dict, got {type(input_data).__name__}")
    
    @property
    def service_type(self) -> str:
        """Get the service type"""
        return "chat"
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Get a list of available models for this service
        
        Returns:
            List of model names
        """
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """
        Get the default model for this service
        
        Returns:
            Default model name
        """
        pass


class EmbeddingService(AIService):
    """
    Base class for embedding-based AI services
    
    Provides a standardized interface for embedding generation services.
    """
    
    def __init__(self, provider: Optional[str] = None, **config):
        """
        Initialize the embedding service
        
        Args:
            provider: Name of the underlying provider (if applicable)
            **config: Additional configuration options
        """
        self.provider = provider
        self.config = config
    
    @abstractmethod
    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **options
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text
        
        Args:
            text: The text or list of texts to embed
            model: The model to use (None for default)
            **options: Additional provider-specific options
            
        Returns:
            The embedding vector(s)
        """
        pass
    
    async def process(self, input_data: Union[str, List[str], Dict[str, Any]], **options) -> Union[List[float], List[List[float]]]:
        """
        Process the input text and return embeddings
        
        Args:
            input_data: Text, list of texts, or dictionary with text and options
            **options: Additional options for processing
            
        Returns:
            The embedding vector(s)
        """
        if isinstance(input_data, (str, list)):
            return await self.embed(text=input_data, **options)
        elif isinstance(input_data, dict):
            text = input_data.pop("text")
            merged_options = {**input_data, **options}
            return await self.embed(text=text, **merged_options)
        else:
            raise TypeError(f"Expected str, list, or dict, got {type(input_data).__name__}")
    
    @property
    def service_type(self) -> str:
        """Get the service type"""
        return "embedding"
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """
        Get a list of available models for this service
        
        Returns:
            List of model names
        """
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """
        Get the default model for this service
        
        Returns:
            Default model name
        """
        pass


class AgentService(AIService):
    """
    Base class for agent-based AI services
    
    Provides a standardized interface for agents that can perform
    multi-step reasoning or actions.
    """
    
    def __init__(self, provider: Optional[str] = None, services: Optional[List[AIService]] = None, **config):
        """
        Initialize the agent service
        
        Args:
            provider: Name of the underlying provider (if applicable)
            services: List of services the agent can use
            **config: Additional configuration options
        """
        self.provider = provider
        self.services = services or []
        self.config = config
    
    @abstractmethod
    async def run(
        self,
        task: str,
        **options
    ) -> Any:
        """
        Run the agent on a task
        
        Args:
            task: The task description or instruction
            **options: Additional agent-specific options
            
        Returns:
            The result of the agent's execution
        """
        pass
    
    async def process(self, input_data: Union[str, Dict[str, Any]], **options) -> Any:
        """
        Process the input task and return the agent's result
        
        Args:
            input_data: Either a string task or a dictionary with task and options
            **options: Additional options for processing
            
        Returns:
            The agent's execution result
        """
        if isinstance(input_data, str):
            return await self.run(task=input_data, **options)
        elif isinstance(input_data, dict):
            task = input_data.pop("task")
            merged_options = {**input_data, **options}
            return await self.run(task=task, **merged_options)
        else:
            raise TypeError(f"Expected str or dict, got {type(input_data).__name__}")
    
    @property
    def service_type(self) -> str:
        """Get the service type"""
        return "agent" 