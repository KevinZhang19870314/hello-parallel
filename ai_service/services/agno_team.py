"""
Agno Team service implementation for multi-agent orchestration
"""
import os
from typing import Any, List, Dict, Optional, Union

from ai_service.base import AgentService
from ai_service.registry import register_service
from ai_service.services.agno_agent import AgnoAgentService

# Try to import agno, but don't fail if it's not installed
try:
    import agno
    from agno.team.team import Team
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from agno.models.anthropic import Claude
    from agno.tools import tool
    from agno.tools.reasoning import ReasoningTools
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    Team = object  # type placeholder
    Agent = object  # type placeholder
    tool = lambda *args, **kwargs: lambda x: x  # type placeholder


class AgnoTeamService(AgentService):
    """
    Multi-agent team service implementation using Agno Teams
    """
    
    def __init__(
        self,
        members: Optional[List[Agent]] = None,
        member_services: Optional[List[AgnoAgentService]] = None,
        mode: str = "coordinate",
        model: Optional[Any] = None,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        provider: Optional[str] = "openai",
        team: Optional[Team] = None,
        name: str = "AI Service Team",
        instructions: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        add_reasoning: bool = True,
        show_members_responses: bool = True,
        enable_agentic_context: bool = True,
        verbose: bool = False,
        markdown: bool = True,
        show_tool_calls: bool = False,
        **config
    ):
        """
        Initialize the Agno team service
        
        Args:
            members: List of Agno Agent instances 
            member_services: List of AgnoAgentService instances to extract agents from
            mode: Team operation mode ("route", "coordinate", or "collaborate")
            model: Pre-configured Agno model for the team leader
            model_name: Model name to use for the team leader if creating a new model
            api_key: API key for the model's service
            provider: Provider name ("openai" or "anthropic")
            team: Pre-created Agno Team instance to use directly
            name: Name of the team
            instructions: List of instructions for the team
            tools: List of tools for the team leader
            add_reasoning: Whether to add reasoning tools to the team leader
            show_members_responses: Whether to show member responses
            enable_agentic_context: Whether to enable agentic context
            verbose: Whether to enable verbose output
            markdown: Whether to use markdown formatting
            show_tool_calls: Whether to show tool calls in the output
            **config: Additional configuration options
        """
        super().__init__(provider=provider, **config)
        
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package is required to use AgnoTeamService. "
                "Please install it with `pip install agno`."
            )
        
        # Use pre-created team if provided
        if team is not None:
            self.team = team
            return
        
        # Get member agents
        self.agents = []
        if members is not None:
            self.agents.extend(members)
        if member_services is not None:
            for service in member_services:
                if hasattr(service, 'agent'):
                    self.agents.append(service.agent)
        
        if not self.agents:
            raise ValueError("At least one agent must be provided")
        
        # Prepare tools for the team leader
        self.tools = tools or []
        
        # Add reasoning tools if requested
        if add_reasoning and not any(isinstance(t, ReasoningTools) for t in self.tools):
            self.tools.append(ReasoningTools(add_instructions=True))
        
        # Setup the model for the team leader based on provider
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
        
        # Create the team
        self.team = Team(
            name=name,
            mode=mode,
            model=self.model,
            members=self.agents,
            tools=self.tools,
            instructions=instructions,
            show_members_responses=show_members_responses,
            enable_agentic_context=enable_agentic_context,
            markdown=markdown,
            debug_mode=verbose,
            show_tool_calls=show_tool_calls,
            **config
        )
    
    async def run(self, task: str, **options) -> Any:
        """
        Run the team on a task
        
        Args:
            task: The task description or instruction
            **options: Additional team-specific options
            
        Returns:
            The result of the team's execution
        """
        try:
            # Run the team async
            response = await self.team.arun(task, **options)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_member(self, member: Agent) -> None:
        """
        Add a new member to the team
        
        Args:
            member: The Agno Agent to add
        """
        self.agents.append(member)
        # Update the team members
        self.team.members = self.agents


# Register the service with the registry if Agno is available
if AGNO_AVAILABLE:
    register_service("agent", "agno_team", AgnoTeamService) 