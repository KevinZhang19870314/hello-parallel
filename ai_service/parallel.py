"""
Parallel processing functionality for AI services
"""
import asyncio
from typing import List, Dict, Any, TypeVar, Generic, Optional, Callable, Tuple

from ai_service.base import AIService, ChatService

# Type variables for input and output
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')
T_Service = TypeVar('T_Service', bound=AIService)


async def parallel_process(
    service: AIService,
    inputs: List[T_Input],
    max_concurrency: Optional[int] = None,
    callback: Optional[Callable[[int, T_Input, Any], None]] = None,
    error_handler: Optional[Callable[[int, T_Input, Exception], Any]] = None,
    **options
) -> List[Any]:
    """
    Process multiple inputs in parallel using a single AI service
    
    Args:
        service: The AI service to use (instance)
        inputs: List of inputs to process
        max_concurrency: Maximum number of concurrent tasks (None for no limit)
        callback: Optional callback function for each completed result
        error_handler: Optional error handler for failed tasks
        **options: Additional options to pass to the service
        
    Returns:
        List of results in the same order as the inputs
    """
    results = [None] * len(inputs)
    
    # Define the semaphore for concurrency control if needed
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def process_item(index: int, input_data: T_Input) -> Tuple[int, Any]:
        """Process a single input with concurrency control"""
        try:
            if semaphore:
                async with semaphore:
                    result = await service.process(input_data, **options)
            else:
                result = await service.process(input_data, **options)
                
            # Call the callback if provided
            if callback:
                callback(index, input_data, result)
                
            return index, result
        except Exception as exc:
            # Handle the error if a handler is provided
            if error_handler:
                result = error_handler(index, input_data, exc)
                return index, result
            else:
                # Re-raise the exception if no handler
                raise
    
    # Create tasks for all inputs
    tasks = [process_item(i, input_data) for i, input_data in enumerate(inputs)]
    
    # Run all tasks and collect results
    for index, result in await asyncio.gather(*tasks):
        results[index] = result
    
    return results


async def parallel_chat(
    service: ChatService,
    prompts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    callback: Optional[Callable[[int, str, str], None]] = None,
    **options
) -> List[str]:
    """
    Send multiple prompts to a chat service concurrently
    
    Args:
        service: A ChatService instance
        prompts: List of prompts to send
        system_prompt: System message to set behavior
        model: Model to use (None for default)
        temperature: Controls randomness
        max_tokens: Maximum tokens to generate
        max_concurrency: Maximum number of concurrent requests
        callback: Optional callback function for completed responses
        **options: Additional options for the chat service
        
    Returns:
        List of responses in the same order as the prompts
    """
    # Prepare processing options
    chat_options = {
        "system_prompt": system_prompt,
        "model": model,
        "temperature": temperature,
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
        **options
    }
    
    # Run parallel processing
    return await parallel_process(
        service=service,
        inputs=prompts,
        max_concurrency=max_concurrency,
        callback=callback,
        **chat_options
    )


async def batch_process(
    services: Dict[str, AIService],
    inputs: Dict[str, List[Any]],
    max_concurrency: Optional[int] = None,
    **options
) -> Dict[str, List[Any]]:
    """
    Process multiple inputs with multiple services in parallel
    
    Args:
        services: Dictionary mapping service keys to AIService instances
        inputs: Dictionary mapping service keys to lists of inputs
        max_concurrency: Maximum number of concurrent tasks per service
        **options: Additional options to pass to the services
        
    Returns:
        Dictionary mapping service keys to lists of results
    """
    results = {}
    tasks = []
    
    # Create tasks for each service and its inputs
    for service_key, service in services.items():
        if service_key in inputs:
            service_inputs = inputs[service_key]
            
            # Create a task for this service's batch
            task = asyncio.create_task(
                parallel_process(
                    service=service,
                    inputs=service_inputs,
                    max_concurrency=max_concurrency,
                    **options
                )
            )
            
            tasks.append((service_key, task))
    
    # Wait for all tasks to complete and collect results
    for service_key, task in tasks:
        results[service_key] = await task
    
    return results 