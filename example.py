"""
AI Service Framework 使用示例
"""
import asyncio
import os
import time
from dotenv import load_dotenv

from ai_service import (
    parallel_chat,
    parallel_process,
    batch_process,
    # 直接客户端适配器函数
    chat_with_openai,
    embed_with_openai,
    process_with_openai,
    process_with_anthropic,
    # Agno适配器
    with_agno_agent,
    with_agno_team,
    process_with_agno
)

# 导入直接服务类
from ai_service.services.openai_chat import OpenAIChatService
from ai_service.services.openai_embedding import OpenAIEmbeddingService
from ai_service.services.anthropic_chat import AnthropicChatService
from ai_service.services.langchain_agent import LangChainAgentService
from ai_service.services.agno_agent import AgnoAgentService
from ai_service.services.agno_team import AgnoTeamService

# 加载环境变量
load_dotenv()

# 用于接收结果的回调函数
def print_result(index, input_data, result):
    """打印单个请求的结果"""
    print(f"\n结果 {index+1}:")
    if isinstance(input_data, str):
        print(f"输入: {input_data}")
    print(f"输出: {result}")

async def example_single_chat():
    """单个聊天服务的示例"""
    print("\n=== 示例1: 使用OpenAI聊天服务 ===")
    
    try:
        # 直接创建聊天服务实例
        api_key = os.getenv("OPENAI_API_KEY")
        chat_service = OpenAIChatService(api_key=api_key)
        
        # 单个请求
        start_time = time.time()
        response = await chat_service.process("解释什么是人工智能")
        
        print(f"响应: {response}")
        print(f"耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_parallel_chat():
    """并行聊天请求的示例"""
    print("\n=== 示例2: 并行聊天请求 ===")
    
    # 准备多个提示
    prompts = [
        "Python是什么编程语言？",
        "什么是机器学习？",
        "解释神经网络的基本原理",
        "什么是深度学习？"
    ]
    
    try:
        # 直接创建聊天服务实例
        api_key = os.getenv("OPENAI_API_KEY")
        chat_service = OpenAIChatService(api_key=api_key)
        
        # 使用服务实例进行并行处理
        start_time = time.time()
        responses = await parallel_chat(
            service=chat_service,  # 直接使用服务实例
            prompts=prompts,
            system_prompt="你是一位专业的AI老师，用简洁的语言回答技术问题",
            temperature=0.7,
            max_concurrency=3,  # 限制最大并发数为3
            callback=print_result  # 使用回调函数接收结果
        )
        
        print(f"\n所有 {len(responses)} 个响应耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_embedding():
    """嵌入服务示例"""
    print("\n=== 示例3: 文本嵌入 ===")
    
    try:
        # 直接创建嵌入服务实例
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_service = OpenAIEmbeddingService(api_key=api_key)
        
        # 生成单个文本的嵌入
        text = "人工智能是计算机科学的一个分支"
        start_time = time.time()
        
        embedding = await embedding_service.process(text)
        
        # 显示嵌入向量的一部分
        vector_preview = str(embedding[:5]) + "..." if embedding else "[]"
        print(f"嵌入向量 (前5个元素): {vector_preview}")
        print(f"向量维度: {len(embedding)}")
        print(f"耗时: {time.time() - start_time:.2f} 秒")
        
        # 生成多个文本的嵌入
        texts = [
            "人工智能正在改变世界",
            "机器学习是AI的一个子领域",
            "深度学习基于神经网络"
        ]
        
        start_time = time.time()
        embeddings = await parallel_process(
            service=embedding_service,  # 直接使用已创建的服务实例
            inputs=texts
        )
        
        print(f"\n处理了 {len(embeddings)} 个文本嵌入，耗时: {time.time() - start_time:.2f} 秒")
        for i, emb in enumerate(embeddings):
            print(f"文本 {i+1} 嵌入维度: {len(emb)}")
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_multiple_services():
    """使用多种服务的示例"""
    print("\n=== 示例4: 使用多种服务 ===")
    
    try:
        # 获取API密钥
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # 直接创建不同类型的服务实例
        services = {
            "chat_openai": OpenAIChatService(api_key=openai_api_key),
            "embed_openai": OpenAIEmbeddingService(api_key=openai_api_key)
        }
        
        # 尝试创建Anthropic聊天服务（如果可用）
        try:
            if anthropic_api_key:
                services["chat_anthropic"] = AnthropicChatService(api_key=anthropic_api_key)
                print("✓ Anthropic服务已创建")
            else:
                print("✗ Anthropic服务不可用: 未找到API密钥")
        except Exception as e:
            print(f"✗ Anthropic服务不可用: {str(e)}")
        
        # 尝试创建LangChain代理服务（如果可用）
        try:
            services["agent_langchain"] = LangChainAgentService(api_key=openai_api_key)
            print("✓ LangChain代理服务已创建")
        except Exception as e:
            print(f"✗ LangChain代理服务不可用: {str(e)}")
        
        # 准备不同类型的输入
        inputs = {
            "chat_openai": [
                "解释量子计算的基本原理", 
                "什么是区块链技术？"
            ],
            "embed_openai": [
                "将这个句子转换为向量表示"
            ]
        }
        
        # 如果Anthropic服务可用，添加输入
        if "chat_anthropic" in services:
            inputs["chat_anthropic"] = ["解释大型语言模型的工作原理"]
        
        # 如果LangChain代理服务可用，添加输入
        if "agent_langchain" in services:
            inputs["agent_langchain"] = ["分析2023年AI领域的主要进展"]
        
        # 使用批处理同时处理所有服务和输入
        start_time = time.time()
        results = await batch_process(
            services=services,  # 传递服务实例字典
            inputs=inputs,
            max_concurrency=2  # 每个服务最多2个并发请求
        )
        
        print(f"\n所有服务处理完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 显示结果
        for service_key, service_results in results.items():
            print(f"\n== {service_key} 结果 ==")
            for i, result in enumerate(service_results):
                result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                print(f"结果 {i+1}: {result_preview}")
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_using_external_instances():
    """使用外部已创建实例的示例"""
    print("\n=== 示例5: 使用外部已创建的服务实例 ===")
    
    try:
        # 创建外部OpenAI客户端
        from openai import AsyncOpenAI
        
        # 用户的OpenAI客户端配置
        api_key = os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)
        
        # 准备测试数据
        prompts = [
            "简要总结人工智能的历史",
            "解释自然语言处理的应用场景"
        ]
        
        texts_to_embed = [
            "这是一个用于嵌入的测试句子",
            "向量表示可以用于相似度计算"
        ]
        
        # 方法1: 使用简化API直接使用客户端进行聊天
        print("\n方法1: 使用简化API直接聊天")
        start_time = time.time()
        chat_responses = await chat_with_openai(
            client=client,
            prompts=prompts,
            system_prompt="你是一个简洁的AI助手",
            temperature=0.5,
            max_concurrency=2
        )
        print(f"聊天完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 显示聊天结果
        for i, response in enumerate(chat_responses):
            print(f"\n提示 {i+1}: {prompts[i]}")
            print(f"响应: {response[:100]}..." if len(response) > 100 else f"响应: {response}")
        
        # 方法2: 使用简化API生成嵌入
        print("\n方法2: 使用简化API生成嵌入")
        start_time = time.time()
        embeddings = await embed_with_openai(
            client=client,
            texts=texts_to_embed,
            max_concurrency=2
        )
        print(f"嵌入生成完成，耗时: {time.time() - start_time:.2f} 秒")
        for i, emb in enumerate(embeddings):
            print(f"文本 {i+1} 嵌入维度: {len(emb)}")
        
        # 方法3: 使用通用处理函数处理聊天
        print("\n方法3: 使用通用处理函数处理聊天")
        start_time = time.time()
        general_responses = await process_with_openai(
            client=client,
            inputs=["通用处理函数示例查询"],
            service_type="chat",
            system_prompt="你是一个帮助回答问题的助手"
        )
        print(f"处理完成，耗时: {time.time() - start_time:.2f} 秒")
        print(f"响应: {general_responses[0][:100]}..." if len(general_responses[0]) > 100 else f"响应: {general_responses[0]}")
        
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_using_langchain_instances():
    """使用外部LangChain实例的示例"""
    print("\n=== 示例6: 使用外部LangChain组件 ===")
    
    try:
        # 导入需要的包
        from langchain_core.tools import Tool
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate
        
        # 创建一个简单的计算器工具
        def calculator(expression: str) -> str:
            """计算数学表达式"""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"计算错误: {str(e)}"
        
        calculator_tool = Tool(
            name="Calculator",
            func=calculator,
            description="用于计算数学表达式，如：1 + 2、3 * 4"
        )
        
        # 创建LangChain模型
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key
        )
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的AI助手，可以回答问题并使用工具解决问题。"),
            ("human", "{input}")
        ])
        
        # 创建代理和执行器
        agent = create_react_agent(llm, [calculator_tool], prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=[calculator_tool],
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 方式1：使用预创建的LLM
        print("\n方式1: 使用预创建的LLM")
        langchain_service1 = LangChainAgentService(
            llm=llm,
            tools=[calculator_tool]
        )
        
        # 方式2：使用预创建的Agent执行器
        print("\n方式2: 使用预创建的Agent执行器")
        langchain_service2 = LangChainAgentService(
            agent_executor=agent_executor
        )
        
        # 准备测试任务
        tasks = [
            "计算25乘以4的结果是多少?",
            "计算100除以5加上20的结果"
        ]
        
        # 使用服务1执行任务
        print("\n使用预创建LLM的服务执行任务:")
        start_time = time.time()
        result1 = await langchain_service1.process(tasks[0])
        print(f"任务: {tasks[0]}")
        print(f"结果: {result1}")
        print(f"耗时: {time.time() - start_time:.2f} 秒")
        
        # 使用服务2执行任务
        print("\n使用预创建Agent执行器的服务执行任务:")
        start_time = time.time()
        result2 = await langchain_service2.process(tasks[1])
        print(f"任务: {tasks[1]}")
        print(f"结果: {result2}")
        print(f"耗时: {time.time() - start_time:.2f} 秒")
        
        # 使用并行处理
        print("\n并行处理LangChain代理任务:")
        start_time = time.time()
        results = await parallel_process(
            service=langchain_service1,
            inputs=tasks,
            max_concurrency=2
        )
        print(f"并行处理完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 显示结果
        for i, result in enumerate(results):
            print(f"\n任务 {i+1}: {tasks[i]}")
            print(f"结果: {result}")
            
    except Exception as e:
        print(f"错误: {str(e)}")

async def example_agno_multi_agent():
    """使用多个Agno代理并行处理任务的示例"""
    print("\n=== 示例7: 使用Agno多代理处理 ===")
    
    try:
        # 检查是否安装了agno库
        try:
            import agno
            from agno.agent import Agent
            from agno.models.openai import OpenAIChat
            from agno.models.anthropic import Claude
            from agno.tools import tool
            from agno.tools.reasoning import ReasoningTools
            from agno.tools.yfinance import YFinanceTools
            from agno.tools.duckduckgo import DuckDuckGoTools
            from agno.team.team import Team
        except ImportError as e:
            print(f"<UNK>: {str(e)}")
            print("需要安装agno库才能运行此示例，请使用 pip install agno yfinance duckduckgo-search")
            return
            
        # 获取API密钥
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_api_key:
            print("需要设置OPENAI_API_KEY环境变量")
            return
            
        print("1. 创建和配置多个Agno Agent")
        
        # 创建一个计算工具
        @tool(show_result=True, stop_after_tool_call=False)
        def calculator(expression: str) -> str:
            """计算数学表达式"""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"计算错误: {str(e)}"
        
        # 创建多个代理服务
        finance_agent = AgnoAgentService(
            api_key=openai_api_key,
            provider="openai",
            model_name="gpt-3.5-turbo",
            tools=[YFinanceTools(
                stock_price=True, 
                company_info=True,
                show_result_tools=["get_stock_price", "get_company_info"],
                stop_after_tool_call_tools=[],
            )],
            instructions=["使用表格显示数据", "包含数据来源"],
            name="金融代理",
            add_reasoning=True,
            show_tool_calls=True
        )
        
        research_agent = AgnoAgentService(
            api_key=openai_api_key,
            provider="openai",
            model_name="gpt-3.5-turbo",
            tools=[DuckDuckGoTools(
                show_result_tools=["duckduckgo_search"],
                stop_after_tool_call_tools=[]
            )],
            instructions=["总是包含信息来源", "回答应当简明扼要"],
            name="研究代理",
            add_reasoning=True,
            show_tool_calls=True
        )
        
        math_agent = AgnoAgentService(
            api_key=openai_api_key,
            provider="openai",
            model_name="gpt-3.5-turbo",
            tools=[calculator],
            instructions=["只返回计算结果", "不需要解释过程"],
            name="计算代理",
            add_reasoning=False,
            show_tool_calls=True
        )
        
        # 如果有Anthropic API密钥，创建Claude代理
        if anthropic_api_key:
            claude_agent = AgnoAgentService(
                api_key=anthropic_api_key,
                provider="anthropic",
                model_name="claude-3-haiku-20240307",
                tools=[],
                instructions=["回答应当全面且有深度", "使用友好的语气"],
                name="Claude代理"
            )
            
        print("2. 准备并行处理的任务")
        
        # 准备任务
        finance_tasks = [
            "获取苹果(AAPL)的当前股价",
            "获取特斯拉(TSLA)的基本公司信息"
        ]
        
        research_tasks = [
            "搜索最新的人工智能研究进展",
            "查找关于量子计算的最新新闻"
        ]
        
        math_tasks = [
            "计算 (123 * 456) / 789",
            "计算 sqrt(225) + 15"
        ]
        
        claude_tasks = [
            "解释大型语言模型的工作原理",
            "探讨人工智能的伦理问题"
        ]
        
        # 配置服务和任务映射
        services = {
            "finance": finance_agent,
            "research": research_agent,
            "math": math_agent
        }
        
        tasks = {
            "finance": finance_tasks,
            "research": research_tasks,
            "math": math_tasks
        }
        
        # 如果有Claude代理，添加到服务和任务映射
        if anthropic_api_key:
            services["claude"] = claude_agent
            tasks["claude"] = claude_tasks
            
        print("3. 使用batch_process并行处理多个代理的多个任务")
        
        # 并行处理所有代理的所有任务
        start_time = time.time()
        results = await batch_process(
            services=services,
            inputs=tasks,
            max_concurrency=2  # 每个代理最多同时处理2个任务
        )
        
        # 显示结果
        print(f"\n所有代理任务完成，总耗时: {time.time() - start_time:.2f}秒")
        for agent_name, agent_results in results.items():
            print(f"\n== {agent_name} 结果 ==")
            for i, (task, result) in enumerate(zip(tasks[agent_name], agent_results)):
                print(f"任务 {i+1}: {task}")
                result_preview = str(result)[:150]
                if len(str(result)) > 150:
                    result_preview += "..."
                print(f"结果: {result_preview}")
                print("-" * 50)
                
        print("\n4. 创建和使用Agno Team（多代理团队）")
        
        # 如果有足够的代理，创建一个团队
        if len(services) >= 2:
            # 从服务中提取代理
            member_services = [services["finance"], services["research"]]
            if "claude" in services:
                member_services.append(services["claude"])
                
            # 创建团队服务
            team_service = AgnoTeamService(
                member_services=member_services,
                api_key=openai_api_key,
                provider="openai",
                model_name="gpt-4", # 使用更强大的模型作为团队领导
                mode="coordinate",
                name="研究分析团队",
                instructions=[
                    "综合所有成员的回答提供全面分析",
                    "使用表格整理数据",
                    "包含所有信息来源"
                ],
                show_members_responses=True,
                show_tool_calls=True
            )
            
            # 准备团队任务
            team_tasks = [
                "分析2023年科技行业的主要趋势，包括AI发展和主要公司(如AAPL, MSFT)的表现",
                "研究量子计算领域的最新进展及其对加密货币市场的潜在影响"
            ]
            
            # 运行团队任务
            print("开始处理团队任务...")
            start_time = time.time()
            team_results = await parallel_process(
                service=team_service,
                inputs=team_tasks,
                max_concurrency=1
            )
            
            # 显示团队结果
            print(f"\n团队任务完成，耗时: {time.time() - start_time:.2f}秒")
            for i, (task, result) in enumerate(zip(team_tasks, team_results)):
                print(f"\n团队任务 {i+1}: {task}")
                result_preview = str(result)[:200]
                if len(str(result)) > 200:
                    result_preview += "..."
                print(f"团队结果: {result_preview}")
                print("=" * 70)
        
    except Exception as e:
        import traceback
        print(f"运行Agno多代理示例时出错: {str(e)}")
        traceback.print_exc()

async def main():
    """运行所有示例"""
    print("AI Service Framework 示例")
    
    # 示例1: 单个聊天请求
    await example_single_chat()
    
    # 示例2: 并行聊天请求
    await example_parallel_chat()
    
    # 示例3: 文本嵌入
    await example_embedding()
    
    # 示例4: 使用多种服务
    await example_multiple_services()
    
    # 示例5: 使用外部已创建的服务实例
    await example_using_external_instances()
    
    # 示例6: 使用外部LangChain组件
    await example_using_langchain_instances()
    
    # 示例7: 使用Agno多代理
    await example_agno_multi_agent()

if __name__ == "__main__":
    # 运行主异步函数
    asyncio.run(main()) 