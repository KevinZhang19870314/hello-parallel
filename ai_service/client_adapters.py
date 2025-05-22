"""
Client adapters for simplified integration with external AI clients
"""
from typing import List, Dict, Any, Optional, Callable, TypeVar, Awaitable

# Type variables
T = TypeVar('T')
U = TypeVar('U')

# OpenAI adapters
def with_openai_client(client):
    """
    Create service instances using an existing OpenAI client
    
    Args:
        client: Pre-created AsyncOpenAI client instance
        
    Returns:
        A dict with chat and embedding service instances
    """
    from ai_service.services.openai_chat import OpenAIChatService
    from ai_service.services.openai_embedding import OpenAIEmbeddingService
    
    return {
        "chat": OpenAIChatService(client=client),
        "embedding": OpenAIEmbeddingService(client=client)
    }

# Anthropic adapters
def with_anthropic_client(client):
    """
    Create a chat service using an existing Anthropic client
    
    Args:
        client: Pre-created AsyncAnthropic client instance
        
    Returns:
        An AnthropicChatService instance
    """
    from ai_service.services.anthropic_chat import AnthropicChatService
    return AnthropicChatService(client=client)

# LangChain adapters
def with_langchain_executor(agent_executor):
    """
    Create an agent service using a pre-created LangChain agent executor
    
    Args:
        agent_executor: Pre-created LangChain AgentExecutor instance
        
    Returns:
        A LangChainAgentService instance
    """
    from ai_service.services.langchain_agent import LangChainAgentService
    return LangChainAgentService(agent_executor=agent_executor)

def with_langchain_llm(llm, tools=None):
    """
    Create an agent service using a pre-created LangChain language model
    
    Args:
        llm: Pre-created LangChain language model
        tools: Optional list of LangChain tools
        
    Returns:
        A LangChainAgentService instance
    """
    from ai_service.services.langchain_agent import LangChainAgentService
    return LangChainAgentService(llm=llm, tools=tools)

# Agno adapters
def with_agno_agent(agent):
    """
    Create an agent service using a pre-created Agno agent
    
    Args:
        agent: Pre-created Agno Agent instance
        
    Returns:
        An AgnoAgentService instance
    """
    from ai_service.services.agno_agent import AgnoAgentService
    return AgnoAgentService(agent=agent)

def with_agno_team(team):
    """
    Create a team service using a pre-created Agno team
    
    Args:
        team: Pre-created Agno Team instance
        
    Returns:
        An AgnoTeamService instance
    """
    from ai_service.services.agno_team import AgnoTeamService
    return AgnoTeamService(team=team)

# Simplified API functions - directly work with clients without creating service instances
async def process_with_openai(client, inputs, *, 
                             service_type="chat", 
                             max_concurrency=None, 
                             **options):
    """
    Process inputs directly with OpenAI client
    
    Args:
        client: Pre-created AsyncOpenAI client instance
        inputs: List of inputs to process
        service_type: Type of service to use ("chat" or "embedding")
        max_concurrency: Maximum number of concurrent requests
        **options: Additional options passed to the service
        
    Returns:
        List of results
    """
    from ai_service.parallel import parallel_process
    
    services = with_openai_client(client)
    service = services.get(service_type)
    if not service:
        raise ValueError(f"Invalid service type: {service_type}")
    
    return await parallel_process(
        service=service,
        inputs=inputs,
        max_concurrency=max_concurrency,
        **options
    )

async def chat_with_openai(client, prompts, *, 
                           system_prompt="You are a helpful assistant.", 
                           model=None, 
                           temperature=0.7,
                           max_tokens=None,
                           max_concurrency=None,
                           callback=None,
                           **options):
    """
    Send multiple chat prompts to OpenAI directly
    
    Args:
        client: Pre-created AsyncOpenAI client instance
        prompts: List of prompts to process
        system_prompt: System message for chat
        model: Model to use
        temperature: Controls randomness
        max_tokens: Maximum tokens to generate
        max_concurrency: Maximum number of concurrent requests
        callback: Optional callback function
        **options: Additional options
        
    Returns:
        List of responses
    """
    from ai_service.parallel import parallel_chat
    
    chat_service = with_openai_client(client)["chat"]
    return await parallel_chat(
        service=chat_service,
        prompts=prompts,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
        callback=callback,
        **options
    )

async def embed_with_openai(client, texts, *, 
                            model=None, 
                            max_concurrency=None,
                            **options):
    """
    Generate embeddings for multiple texts with OpenAI
    
    Args:
        client: Pre-created AsyncOpenAI client instance
        texts: Single text or list of texts to embed
        model: Model to use
        max_concurrency: Maximum number of concurrent requests
        **options: Additional options
        
    Returns:
        List of embedding vectors
    """
    from ai_service.parallel import parallel_process
    
    embedding_service = with_openai_client(client)["embedding"]
    
    # Handle both single text and list of texts
    single_input = not isinstance(texts, list)
    inputs = [texts] if single_input else texts
    
    embeddings = await parallel_process(
        service=embedding_service,
        inputs=inputs,
        max_concurrency=max_concurrency,
        model=model,
        **options
    )
    
    # Return single embedding for single input
    return embeddings[0] if single_input else embeddings

async def process_with_anthropic(client, prompts, *,
                                system_prompt="You are a helpful assistant.",
                                model=None,
                                temperature=0.7, 
                                max_tokens=None,
                                max_concurrency=None,
                                callback=None,
                                **options):
    """
    Process multiple prompts with Anthropic
    
    Args:
        client: Pre-created AsyncAnthropic client instance
        prompts: List of prompts to process
        system_prompt: System message
        model: Model to use
        temperature: Controls randomness
        max_tokens: Maximum tokens to generate
        max_concurrency: Maximum number of concurrent requests
        callback: Optional callback function
        **options: Additional options
        
    Returns:
        List of responses
    """
    from ai_service.parallel import parallel_chat
    
    chat_service = with_anthropic_client(client)
    return await parallel_chat(
        service=chat_service,
        prompts=prompts,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
        callback=callback,
        **options
    )

async def process_with_agno(agent, inputs, *,
                           max_concurrency=None,
                           callback=None,
                           **options):
    """
    Process multiple inputs with an Agno agent
    
    Args:
        agent: Pre-created Agno Agent instance
        inputs: List of inputs to process
        max_concurrency: Maximum number of concurrent requests
        callback: Optional callback function
        **options: Additional options
        
    Returns:
        List of responses
    """
    from ai_service.parallel import parallel_process
    
    agent_service = with_agno_agent(agent)
    return await parallel_process(
        service=agent_service,
        inputs=inputs,
        max_concurrency=max_concurrency,
        callback=callback,
        **options
    ) 