"""
Agno Agent service implementation
"""
import os
from typing import Any, List, Optional, Dict, Union, Callable

from ai_service.base import AgentService
from ai_service.registry import register_service

# Try to import agno, but don't fail if it's not installed
try:
    import agno
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from agno.models.anthropic import Claude
    from agno.tools import tool
    from agno.tools.reasoning import ReasoningTools
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    Agent = object  # type placeholder
    tool = lambda *args, **kwargs: lambda x: x  # type placeholder


class AgnoAgentService(AgentService):
    """
    Agent service implementation using Agno framework
    """
    
    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        model: Optional[Any] = None,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        provider: Optional[str] = "openai",
        agent: Optional[Agent] = None,
        instructions: Optional[List[str]] = None,
        knowledge: Optional[Any] = None,
        storage: Optional[Any] = None,
        memory: Optional[Any] = None,
        add_reasoning: bool = True,
        verbose: bool = False,
        markdown: bool = True,
        show_tool_calls: bool = False,
        **config
    ):
        """
        Initialize the Agno agent service
        
        Args:
            tools: List of Agno tools for the agent to use
            model: Pre-configured Agno model
            model_name: Model name to use if creating a new model
            api_key: API key for the model's service
            provider: Provider name ("openai" or "anthropic")
            agent: Pre-created Agno agent to use directly
            instructions: List of instructions for the agent
            knowledge: Knowledge base for the agent
            storage: Storage for the agent sessions
            memory: Memory for the agent
            add_reasoning: Whether to add reasoning tools
            verbose: Whether to enable verbose output
            markdown: Whether to use markdown formatting
            show_tool_calls: Whether to show tool calls in the output
            **config: Additional configuration options
        """
        super().__init__(provider=provider, **config)
        
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package is required to use AgnoAgentService. "
                "Please install it with `pip install agno`."
            )
        
        # Use pre-created agent if provided
        if agent is not None:
            self.agent = agent
            return
        
        # Prepare tools
        self.tools = tools or []
        
        # Add reasoning tools if requested
        if add_reasoning and not any(isinstance(t, ReasoningTools) for t in self.tools):
            self.tools.insert(0, ReasoningTools(add_instructions=True))
        
        # Setup the model based on provider
        if model is not None:
            self.model = model
        elif provider.lower() == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided for Anthropic")
            self.model = Claude(id=model_name, api_key=api_key)
        else:  # Default to OpenAI
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided for OpenAI")
            self.model = OpenAIChat(id=model_name, api_key=api_key)
        
        # Create the agent
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            instructions=instructions,
            knowledge=knowledge,
            storage=storage,
            memory=memory,
            markdown=markdown,
            debug_mode=verbose,
            show_tool_calls=show_tool_calls,
            **config
        )
    
    async def run(self, task: str, **options) -> Any:
        """
        Run the agent on a task
        
        Args:
            task: The task description or instruction
            **options: Additional agent-specific options
            
        Returns:
            The result of the agent's execution
        """
        try:
            # Run the agent async
            response = await self.agent.arun(task, **options)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_tool(self, tool_func: Any) -> None:
        """
        Add a new tool to the agent
        
        Args:
            tool_func: The Agno tool function to add
        """
        self.tools.append(tool_func)
        # Update the agent tools
        self.agent.tools = self.tools


# Register the service with the registry if Agno is available
if AGNO_AVAILABLE:
    register_service("agent", "agno", AgnoAgentService) 