# AI Service Framework

灵活、可扩展的AI服务框架，支持并行处理，专为各种AI服务类型设计。

## 特点

- **服务类型抽象**，而非提供商抽象
- 统一接口支持任意类型的AI服务（聊天、嵌入、代理等）
- 异步处理和并行请求处理
- 对多种AI服务类型的内置支持
- 可配置的并发控制
- 用于实时处理的回调机制
- 支持服务扩展和自定义实现
- 完善的错误处理
- 与外部已创建客户端的无缝集成

## 安装

```bash
pip install ai-service-framework
```

## 快速开始

### 聊天服务

```python
import asyncio
import os
from openai import AsyncOpenAI
from ai_service import parallel_chat
from ai_service.services.openai_chat import OpenAIChatService

# 设置API密钥
api_key = os.environ["OPENAI_API_KEY"]

async def main():
    # 直接创建服务实例
    chat_service = OpenAIChatService(api_key=api_key)
    
    # 单个请求
    result = await chat_service.process("解释量子计算。")
    print(result)
    
    # 并行请求
    prompts = [
        "Python是什么编程语言？", 
        "什么是神经网络？",
        "深度学习与传统机器学习有什么区别？"
    ]
    
    results = await parallel_chat(
        service=chat_service,
        prompts=prompts,
        max_tokens=100,
        callback=lambda idx, p, r: print(f"收到回答 {idx+1}: {r[:30]}...")
    )
    
    for prompt, response in zip(prompts, results):
        print(f"问题: {prompt}")
        print(f"回答: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用预创建客户端

```python
import asyncio
import os
from openai import AsyncOpenAI
from ai_service import chat_with_openai, embed_with_openai

async def main():
    # 创建OpenAI客户端
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    
    # 并行聊天请求
    prompts = ["解释人工智能", "什么是机器学习？"]
    responses = await chat_with_openai(
        client=client,
        prompts=prompts,
        system_prompt="你是一个教育专家",
        temperature=0.7,
        max_concurrency=2  # 限制并发
    )
    
    # 文本嵌入请求
    texts = ["人工智能", "机器学习", "深度学习"]
    embeddings = await embed_with_openai(
        client=client,
        texts=texts
    )
    
    print(f"生成了 {len(embeddings)} 个嵌入向量")

if __name__ == "__main__":
    asyncio.run(main())
```

### 使用多种服务

```python
import asyncio
import os
from ai_service import batch_process
from ai_service.services.openai_chat import OpenAIChatService
from ai_service.services.openai_embedding import OpenAIEmbeddingService
from ai_service.services.langchain_agent import LangChainAgentService

async def main():
    # 获取API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 直接创建不同类型的服务实例
    chat = OpenAIChatService(api_key=api_key)
    embedding = OpenAIEmbeddingService(api_key=api_key)
    agent = LangChainAgentService(api_key=api_key)
    
    # 准备不同类型的输入
    inputs = {
        "chat": ["解释机器学习。", "什么是神经网络？"],
        "embedding": ["将这个句子转换为向量"],
        "agent": ["搜索有关2023年人工智能进展的信息。"]
    }
    
    # 将所有服务和输入打包在一起处理
    services = {"chat": chat, "embedding": embedding, "agent": agent}
    
    # 并行处理所有请求
    results = await batch_process(services, inputs)
    
    # 打印结果
    for service_type, service_results in results.items():
        print(f"\n== {service_type.upper()} 结果 ==")
        for i, result in enumerate(service_results):
            print(f"结果 {i+1}: {str(result)[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## 高级用法

### 集成已有的客户端

```python
import asyncio
import os
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from ai_service import (
    with_openai_client,
    with_anthropic_client,
    process_with_openai,
    process_with_anthropic,
    parallel_chat,
    parallel_process
)

async def main():
    # 创建已有的客户端
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # 方法1: 使用简化函数直接处理
    openai_responses = await process_with_openai(
        client=openai_client,
        inputs=["什么是人工智能？", "什么是机器学习？"],
        service_type="chat",  # 或 "embedding"
        system_prompt="你是一位AI教授"
    )
    
    # 方法2: 创建服务实例然后处理
    openai_services = with_openai_client(openai_client)
    chat_service = openai_services["chat"]
    embedding_service = openai_services["embedding"]
    
    # 使用聊天服务
    chat_results = await parallel_chat(
        service=chat_service,
        prompts=["请解释神经网络", "什么是深度学习"],
        system_prompt="简明扼要地回答"
    )
    
    # 使用嵌入服务
    texts = ["人工智能", "机器学习", "深度学习"]
    embedding_results = await parallel_process(
        service=embedding_service,
        inputs=texts
    )
    
    # 使用Anthropic服务
    anthropic_service = with_anthropic_client(anthropic_client)
    claude_results = await parallel_chat(
        service=anthropic_service,
        prompts=["什么是大型语言模型？"],
        system_prompt="你是一个有帮助的助手"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 注册自定义服务

```python
from ai_service.base import AIService
from ai_service.registry import register_service

class MyCustomAIService(AIService):
    """自定义AI服务实现"""
    
    def __init__(self, provider=None, **config):
        self.provider = provider
        self.config = config
        # 初始化自定义服务
        
    async def process(self, input_data, **options):
        # 实现处理逻辑
        return f"处理结果: {input_data}"
        
    @property
    def service_type(self):
        return "custom"

# 如果需要，可以注册服务以便通过工厂函数创建
register_service("custom", "my_service", MyCustomAIService)

# 直接创建服务实例
custom_service = MyCustomAIService(parameter1="value1")
```

### 创建自定义聊天服务

```python
from ai_service.base import ChatService
from ai_service.registry import register_service

class MyCustomChatService(ChatService):
    """自定义聊天服务实现"""
    
    def __init__(self, provider=None, **config):
        super().__init__(provider, **config)
        # 初始化自定义聊天服务
        
    async def generate(self, prompt, system_prompt="你是助手", **options):
        # 实现生成逻辑
        return f"回答: {prompt}"
    
    @property
    def available_models(self):
        return ["my-model-1", "my-model-2"]
    
    @property
    def default_model(self):
        return "my-model-1"

# 如果需要，可以注册服务
register_service("chat", "my_chat", MyCustomChatService)

# 直接创建服务实例
my_chat = MyCustomChatService()
```

### 创建代理服务

```python
from ai_service.base import AgentService
from ai_service.services.openai_chat import OpenAIChatService

class TaskSolverAgent(AgentService):
    """自定义代理服务"""
    
    def __init__(self, api_key=None, provider=None, **config):
        super().__init__(provider, **config)
        # 创建依赖服务
        self.chat = OpenAIChatService(api_key=api_key)
        
    async def run(self, task, **options):
        # 实现任务解决逻辑，可能使用多个服务协同工作
        result = await self.chat.process(f"解决这个任务: {task}")
        return f"代理结果: {result}"
        
    @property
    def service_type(self):
        return "agent"

# 直接创建服务实例
agent = TaskSolverAgent(api_key="your_api_key")
```

## 架构概述

该框架基于以下核心概念:

### 1. 服务类型

不同于以提供商分类，我们按服务类型分类:

- **chat**: 聊天/对话完成服务
- **embedding**: 文本嵌入服务
- **agent**: 代理服务（多步骤推理或动作执行）
- 可扩展更多服务类型

### 2. 抽象基类

- `AIService`: 所有AI服务的基础抽象类
- `ChatService`: 聊天服务的抽象类
- `EmbeddingService`: 嵌入服务的抽象类
- `AgentService`: 代理服务的抽象类

### 3. 并行处理

- `parallel_process(service, inputs, ...)`: 使用单个服务并行处理多个输入
- `parallel_chat(service, prompts, ...)`: 并行处理多个聊天请求
- `batch_process(services, inputs, ...)`: 使用多个服务并行处理多个输入

### 4. 客户端适配器

- `with_openai_client(client)`: 使用预创建的OpenAI客户端创建服务
- `with_anthropic_client(client)`: 使用预创建的Anthropic客户端创建服务
- `with_langchain_executor(agent_executor)`: 使用预创建的LangChain执行器创建服务
- `with_langchain_llm(llm, tools)`: 使用预创建的LangChain语言模型创建服务

### 5. 简化API函数

- `chat_with_openai(client, prompts, ...)`: 使用OpenAI客户端直接进行并行聊天
- `embed_with_openai(client, texts, ...)`: 使用OpenAI客户端直接生成嵌入
- `process_with_openai(client, inputs, ...)`: 使用OpenAI客户端进行通用处理
- `process_with_anthropic(client, prompts, ...)`: 使用Anthropic客户端处理提示

## 支持的服务

当前包含以下预实现服务:

### 聊天服务
- `OpenAIChatService`: 使用OpenAI的GPT模型
- `AnthropicChatService`: 使用Anthropic的Claude模型

### 嵌入服务
- `OpenAIEmbeddingService`: 使用OpenAI的文本嵌入模型

### 代理服务
- `LangChainAgentService`: 基于LangChain框架的代理实现

## 扩展框架

你可以通过以下方式扩展框架:

1. **添加新服务类型**: 继承`AIService`创建新的服务类型抽象类
2. **添加新服务提供商**: 为现有服务类型实现新的提供商
3. **创建自定义代理**: 通过继承`AgentService`并实现`run`方法
4. **集成已有客户端**: 创建新的客户端适配器函数

## 依赖要求

- Python 3.7+
- 异步支持通过Python `asyncio`
- 可选依赖:
  - `openai` 用于OpenAI服务
  - `anthropic` 用于Anthropic服务
  - `langchain` 和 `langchain-openai` 用于LangChain代理

## 贡献

欢迎贡献！请参阅[贡献指南](CONTRIBUTING.md)了解详情。

## 许可证

MIT 