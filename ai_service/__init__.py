"""
AI Service - A flexible framework for interacting with various AI services

This package provides a unified interface for different types of AI services,
with built-in support for parallel processing.
"""

# 服务基类
from ai_service.base import AIService, ChatService, EmbeddingService, AgentService

# 并行处理函数
from ai_service.parallel import parallel_process, parallel_chat, batch_process

# 简化的客户端适配函数
from ai_service.client_adapters import (
    with_openai_client, 
    with_anthropic_client, 
    with_langchain_executor, 
    with_langchain_llm, 
    process_with_openai, 
    chat_with_openai,
    embed_with_openai,
    process_with_anthropic,
    with_agno_agent,
    with_agno_team,
    process_with_agno
)

# 直接导入服务实现，便于用户直接实例化
from ai_service.services.openai_chat import OpenAIChatService
from ai_service.services.openai_embedding import OpenAIEmbeddingService
from ai_service.services.anthropic_chat import AnthropicChatService
from ai_service.services.langchain_agent import LangChainAgentService
from ai_service.services.agno_agent import AgnoAgentService
from ai_service.services.agno_team import AgnoTeamService

# 服务注册与工厂函数 (保留向后兼容性)
from ai_service.registry import register_service, create_service, get_registered_services

# 导入所有服务实现以确保它们被注册 (保留向后兼容性)
import ai_service.services

__all__ = [
    # 基类
    'AIService', 'ChatService', 'EmbeddingService', 'AgentService',
    
    # 并行处理函数
    'parallel_process', 'parallel_chat', 'batch_process',
    
    # 客户端适配器
    'with_openai_client', 'with_anthropic_client', 'with_langchain_executor', 'with_langchain_llm',
    'process_with_openai', 'chat_with_openai', 'embed_with_openai', 'process_with_anthropic',
    'with_agno_agent', 'with_agno_team', 'process_with_agno',
    
    # 服务实现类
    'OpenAIChatService', 'OpenAIEmbeddingService', 'AnthropicChatService', 'LangChainAgentService', 
    'AgnoAgentService', 'AgnoTeamService',
    
    # 服务注册工厂 (保留向后兼容性)
    'register_service', 'create_service', 'get_registered_services'
] 