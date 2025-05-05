# utils/config_manager.py

import os
import json
import logging
import streamlit as st
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("config_manager")

class ConfigManager:
    """
    Centralized configuration management for the Agentic RAG application.
    Handles loading, validation, and access to configuration parameters.
    """
    
    def __init__(self, dotenv_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            dotenv_path (Optional[str]): Path to .env file (if not using default)
        """
        # Load environment variables
        self.load_environment(dotenv_path)
        
        # Initialize configuration state
        self.config = {}
        self.service_status = {}
        
        # Load and validate the configuration
        self.load_config()
        self.validate_config()
    
    def load_environment(self, dotenv_path: Optional[str] = None) -> None:
        """
        Load environment variables from .env file
        
        Args:
            dotenv_path (Optional[str]): Path to .env file
        """
        try:
            if dotenv_path and os.path.exists(dotenv_path):
                load_env_result = load_dotenv(dotenv_path, override=True)
            else:
                load_env_result = load_dotenv(override=True)
                
            if load_env_result:
                logger.info("Loaded environment variables from .env file")
            else:
                logger.warning("No .env file found or no variables loaded")
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
    
    def load_config(self) -> None:
        """Load configuration from all available sources with precedence order."""
        # 1. Default configuration
        self.config = {
            # LLM Configuration
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_temperature": 0,
            "llm_timeout": 30,
            
            # Embedding Configuration
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "embedding_device": "cpu",
            
            # RAG Configuration
            "chunk_size": 500,
            "chunk_overlap": 100,
            "max_retrieved_docs": 5,
            
            # Web Search Configuration
            "web_search_enabled": True,
            "web_search_provider": "tavily",
            "web_search_max_results": 3,
            
            # API Keys (empty by default)
            "openai_api_key": "",
            "tavily_api_key": "",
            
            # Path Configuration
            "default_knowledge_path": "data/default_knowledge.pdf",
            "vector_store_path": "data/vector_store"
        }
        
        # 2. Load from environment variables
        self._load_from_environment()
        
        # 3. Load from config file if it exists
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # 4. Store session state config if available
        if "config" in st.session_state:
            self.config.update(st.session_state.config)
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # API Keys
        if os.environ.get("OPENAI_API_KEY"):
            self.config["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
            
        if os.environ.get("TAVILY_API_KEY"):
            self.config["tavily_api_key"] = os.environ.get("TAVILY_API_KEY")
        
        # LLM Configuration    
        if os.environ.get("LLM_MODEL"):
            self.config["llm_model"] = os.environ.get("LLM_MODEL")
            
        if os.environ.get("LLM_PROVIDER"):
            self.config["llm_provider"] = os.environ.get("LLM_PROVIDER")
        
        # Embedding Configuration
        if os.environ.get("EMBEDDING_MODEL_NAME"):
            self.config["embedding_model"] = os.environ.get("EMBEDDING_MODEL_NAME")
            
        if os.environ.get("EMBEDDING_PROVIDER"):
            self.config["embedding_provider"] = os.environ.get("EMBEDDING_PROVIDER")
            
        if os.environ.get("EMBEDDING_DEVICE"):
            self.config["embedding_device"] = os.environ.get("EMBEDDING_DEVICE")
        
        # RAG Configuration
        if os.environ.get("CHUNK_SIZE"):
            self.config["chunk_size"] = int(os.environ.get("CHUNK_SIZE"))
            
        if os.environ.get("CHUNK_OVERLAP"):
            self.config["chunk_overlap"] = int(os.environ.get("CHUNK_OVERLAP"))
            
        if os.environ.get("MAX_RETRIEVED_DOCS"):
            self.config["max_retrieved_docs"] = int(os.environ.get("MAX_RETRIEVED_DOCS"))
        
        # Web Search Configuration
        if os.environ.get("WEB_SEARCH_MAX_RESULTS"):
            self.config["web_search_max_results"] = int(os.environ.get("WEB_SEARCH_MAX_RESULTS"))
    
    def validate_config(self) -> None:
        """Validate configuration and check API key validity."""
        self.service_status = {
            "openai_api": False,
            "tavily_api": False,
            "embedding_model": False,
            "llm": False
        }
        
        # Validate OpenAI API Key
        openai_key = self.config.get("openai_api_key", "")
        if openai_key and not openai_key.startswith("sk-None") and len(openai_key) > 20:
            self.service_status["openai_api"] = True
        else:
            logger.warning("OpenAI API Key is invalid or not provided")
        
        # Validate Tavily API Key
        tavily_key = self.config.get("tavily_api_key", "")
        if tavily_key and tavily_key != "None" and len(tavily_key) > 10:
            self.service_status["tavily_api"] = True
        else:
            logger.warning("Tavily API Key is invalid or not provided")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with an optional default.
        
        Args:
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value and update session state.
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
        """
        self.config[key] = value
        
        # Update session state if using Streamlit
        if "config" not in st.session_state:
            st.session_state.config = {}
        st.session_state.config[key] = value
    
    def get_service_status(self) -> Dict[str, bool]:
        """
        Get the status of all services.
        
        Returns:
            Dict[str, bool]: Dictionary of service statuses
        """
        return self.service_status
    
    def is_service_available(self, service_name: str) -> bool:
        """
        Check if a specific service is available.
        
        Args:
            service_name (str): Name of the service to check
            
        Returns:
            bool: True if service is available, False otherwise
        """
        return self.service_status.get(service_name, False)
    
    def prompt_for_api_keys(self) -> None:
        """Prompt user for missing API keys using Streamlit UI."""
        st.sidebar.subheader("API Keys")
        
        # OpenAI API Key
        if not self.service_status["openai_api"]:
            openai_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key"
            )
            if openai_key and not openai_key.startswith("sk-None"):
                self.set("openai_api_key", openai_key)
                os.environ["OPENAI_API_KEY"] = openai_key
                self.service_status["openai_api"] = True
                st.sidebar.success("OpenAI API Key set successfully!")
        
        # Tavily API Key
        if not self.service_status["tavily_api"] and self.get("web_search_enabled", True):
            tavily_key = st.sidebar.text_input(
                "Tavily API Key",
                type="password", 
                help="Enter your Tavily API key for web search"
            )
            if tavily_key and tavily_key != "None":
                self.set("tavily_api_key", tavily_key)
                os.environ["TAVILY_API_KEY"] = tavily_key
                self.service_status["tavily_api"] = True
                st.sidebar.success("Tavily API Key set successfully!")
    
    def display_status(self) -> None:
        """Display service status in Streamlit UI."""
        st.sidebar.subheader("System Status")
        
        for service, status in self.service_status.items():
            service_name = service.replace("_", " ").title()
            status_icon = "✅" if status else "❌"
            st.sidebar.markdown(f"- {service_name}: {status_icon}")