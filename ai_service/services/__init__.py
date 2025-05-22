"""
AI service implementations
"""

# Import all service modules to ensure registration code runs
import ai_service.services.openai_chat
import ai_service.services.anthropic_chat
import ai_service.services.openai_embedding
import ai_service.services.langchain_agent

# Also export the classes for direct import
from ai_service.services.openai_chat import OpenAIChatService
from ai_service.services.anthropic_chat import AnthropicChatService
from ai_service.services.openai_embedding import OpenAIEmbeddingService
from ai_service.services.langchain_agent import LangChainAgentService

# More service imports can be added here 