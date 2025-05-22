"""
LangChain Agent service implementation
"""
import os
from typing import Any, List, Optional, Dict, Union

from ai_service.base import AgentService, ChatService
from ai_service.registry import register_service, create_service

# Try to import langchain, but don't fail if it's not installed
try:
    from langchain_core.tools import BaseTool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.language_models import BaseChatModel
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object  # type placeholder
    ChatPromptTemplate = object  # type placeholder
    BaseChatModel = object  # type placeholder
    AgentExecutor = object  # type placeholder


class LangChainAgentService(AgentService):
    """
    Agent service implementation using LangChain
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        chat_service: Optional[ChatService] = None,
        llm: Optional[BaseChatModel] = None,
        agent_executor: Optional[AgentExecutor] = None,
        **config
    ):
        """
        Initialize the LangChain agent service
        
        Args:
            tools: List of LangChain tools for the agent to use
            model: The model to use with the agent
            api_key: API key for the model's service
            provider: Provider name (for registry identification)
            chat_service: Optional chat service to use
            llm: Pre-created LangChain language model to use
            agent_executor: Pre-created LangChain agent executor to use directly
            **config: Additional configuration options
        """
        super().__init__(provider=provider, **config)
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain packages are required to use LangChainAgentService. "
                "Please install them with `pip install langchain langchain-openai`."
            )
        
        # Use pre-created agent executor if provided
        if agent_executor is not None:
            self.agent = agent_executor
            self.tools = agent_executor.tools
            self.llm = None  # Not needed if we have a pre-created executor
            return
            
        # Set up tools
        self.tools = tools or []
        self.model = model
        
        # Set up language model (either use provided one or create a default)
        if llm is not None:
            self.llm = llm
        elif chat_service:
            self.chat_service = chat_service
            # Use the provided chat service (advanced integration would be needed here)
            # This is a placeholder for a custom integration
            self.llm = None  # Direct chat service integration not implemented
        else:
            # Get API key
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided")
            
            # Create a default LangChain model
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.7,
                api_key=self.api_key,
                **config
            )
        
        # Create the agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create a LangChain agent with the configured tools and model"""
        if not self.llm:
            raise ValueError("No language model available for creating agent")
            
        # Create a prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to tools. "
                      "Answer user queries and use tools when necessary."),
            ("human", "{input}")
        ])
        
        # Create the ReAct agent
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        # Create an agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.config.get("verbose", False),
            handle_parsing_errors=True
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
            # Run the agent
            response = await self.agent.ainvoke({"input": task, **options})
            return response.get("output", "Error: No output from agent")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a new tool to the agent
        
        Args:
            tool: The LangChain tool to add
        """
        self.tools.append(tool)
        # Recreate the agent with the new tool
        if hasattr(self, '_create_agent'):
            self.agent = self._create_agent()


# Register the service with the registry if LangChain is available
if LANGCHAIN_AVAILABLE:
    register_service("agent", "langchain", LangChainAgentService) 