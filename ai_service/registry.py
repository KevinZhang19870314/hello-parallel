"""
Service registry and factory functions
"""
from typing import Dict, Type, Any, List, Optional, Tuple

from ai_service.base import AIService

# Registry of service implementations, organized by type and name
# Format: {service_type: {service_name: service_class}}
SERVICE_REGISTRY: Dict[str, Dict[str, Type[AIService]]] = {}


def register_service(service_type: str, name: str, service_class: Type[AIService]) -> None:
    """
    Register a new AI service implementation
    
    Args:
        service_type: The type of service (e.g., 'chat', 'embedding', 'agent')
        name: The name of the service implementation
        service_class: The class implementing the service
    """
    # Initialize the type registry if needed
    if service_type not in SERVICE_REGISTRY:
        SERVICE_REGISTRY[service_type] = {}
    
    # Register the service class
    SERVICE_REGISTRY[service_type][name.lower()] = service_class


def create_service(service_type: str, name: str, **config) -> AIService:
    """
    Create an instance of a registered service
    
    Args:
        service_type: The type of service (e.g., 'chat', 'embedding', 'agent')
        name: The name of the service implementation
        **config: Configuration options for the service
        
    Returns:
        An instance of the requested service
        
    Raises:
        ValueError: If the service type or name is not registered
    """
    # Check if the service type exists
    if service_type not in SERVICE_REGISTRY:
        available_types = ", ".join(SERVICE_REGISTRY.keys()) if SERVICE_REGISTRY else "none"
        raise ValueError(f"Unknown service type: '{service_type}'. Available types: {available_types}")
    
    # Check if the service name exists for this type
    name_lower = name.lower()
    if name_lower not in SERVICE_REGISTRY[service_type]:
        available_services = ", ".join(SERVICE_REGISTRY[service_type].keys())
        raise ValueError(f"Unknown service name: '{name}' for type '{service_type}'. Available services: {available_services}")
    
    # Create and return the service instance
    service_class = SERVICE_REGISTRY[service_type][name_lower]
    return service_class(provider=name, **config)


def get_registered_services() -> Dict[str, List[str]]:
    """
    Get a dictionary of all registered services
    
    Returns:
        A dictionary mapping service types to lists of service names
    """
    return {
        service_type: list(services.keys())
        for service_type, services in SERVICE_REGISTRY.items()
    }


def get_service_info(service_type: Optional[str] = None) -> List[Tuple[str, str, Type[AIService]]]:
    """
    Get detailed information about registered services
    
    Args:
        service_type: Optional filter for a specific service type
        
    Returns:
        List of tuples containing (service_type, service_name, service_class)
    """
    result = []
    
    if service_type:
        if service_type in SERVICE_REGISTRY:
            for name, cls in SERVICE_REGISTRY[service_type].items():
                result.append((service_type, name, cls))
    else:
        for type_name, services in SERVICE_REGISTRY.items():
            for name, cls in services.items():
                result.append((type_name, name, cls))
    
    return result 