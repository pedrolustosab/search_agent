import os
import streamlit as st
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain.callbacks import get_openai_callback
from typing import Dict, Any

class Config:
    """
    Configuration management class
    
    Attributes:
        openai_api_key (str): OpenAI API key
        search_api_key (str): API key for the selected search provider
        search_provider (str): Selected search provider (SerpAPI or Tavily)
        model_name (str): Name of the OpenAI model to use
        max_tokens (int): Maximum number of tokens for model responses
    
    Methods:
        validate_keys(): Validates if all required API keys are present
        llm_settings (property): Returns a dictionary of LLM settings
    """
    def __init__(self):
        self.openai_api_key = None
        self.search_api_key = None
        self.search_provider = None
        self.model_name = "gpt-3.5-turbo-0125"
        self.max_tokens = 2000
        
    def validate_keys(self) -> bool:
        """Validate if all required API keys are present"""
        if not self.openai_api_key:
            st.sidebar.error("OpenAI API Key is required")
            return False
        if not self.search_api_key:
            st.sidebar.error(f"{self.search_provider} API Key is required")
            return False
        return True
            
    @property
    def llm_settings(self):
        return {
            'temperature': 0.3,
            'openai_api_key': self.openai_api_key,
            'model_name': self.model_name,
            'max_tokens': self.max_tokens
        }

class AIAgent:
    """
    AI Agent handling class
    
    Attributes:
        config (Config): Configuration object
        llm: Language model instance
        agent: Initialized agent with tools
    
    Methods:
        setup_agent(): Initializes the LLM and agent with tools
        process_query(query: str): Processes a query and returns the response with token usage
    """
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.agent = None
        self.setup_agent()
        
    def setup_agent(self):
        """Initialize the LLM and agent with tools"""
        try:
            self.llm = ChatOpenAI(**self.config.llm_settings)
            
            # Initialize tools based on provider
            if self.config.search_provider == "SerpAPI":
                os.environ['SERPAPI_API_KEY'] = self.config.search_api_key
                tools = load_tools(
                    ["serpapi", "llm-math"],
                    llm=self.llm
                )
            else:  # Tavily
                os.environ['TAVILY_API_KEY'] = self.config.search_api_key
                
                # Initialize Tavily search tool
                tavily_tool = TavilySearchResults()
                
                # Get math tool
                math_tool = load_tools(["llm-math"], llm=self.llm)[0]
                
                # Combine tools
                tools = [tavily_tool, math_tool]
            
            self.agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                return_intermediate_steps=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize agent: {str(e)}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return the response with token usage"""
        if not query.strip():
            return {"error": "Query cannot be empty"}
            
        try:
            with get_openai_callback() as cb:
                response = self.agent(query)
                token_usage = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "total_cost": f"${cb.total_cost:.4f}"
                }
            
            return {
                "success": True,
                "output": response['output'],
                "intermediate_steps": response.get('intermediate_steps', []),
                "token_usage": token_usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def setup_sidebar() -> Config:
    """
    Setup sidebar with API key inputs and search provider selector
    
    Returns:
        Config: Configuration object with user-provided settings
    """
    st.sidebar.title("Configuration")
    
    config = Config()
    
    # API Keys input
    config.openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        type="password"
    )
    
    # Search provider selector
    config.search_provider = st.sidebar.selectbox(
        "Select Search Provider",
        options=["SerpAPI", "Tavily"],
        index=0
    )
    
    # Dynamic API key input based on selected provider
    if config.search_provider == "SerpAPI":
        config.search_api_key = st.sidebar.text_input(
            "SerpAPI Key",
            value="",
            type="password"
        )
    else:
        config.search_api_key = st.sidebar.text_input(
            "Tavily API Key",
            value="",
            type="password"
        )
    
    # Model settings expander
    with st.sidebar.expander("Model Settings"):
        config.model_name = st.selectbox(
            "Model",
            options=["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"],
            index=0
        )
        config.max_tokens = st.slider(
            "Max Response Tokens",
            min_value=256,
            max_value=4096,
            value=2000,
            help="Maximum number of tokens in the response"
        )
    
    return config

def init_session_state(config: Config):
    """
    Initialize or update session state
    
    Args:
        config (Config): Configuration object
    """
    try:
        if 'agent' not in st.session_state or \
           st.session_state.get('last_config') != vars(config):
            st.session_state.agent = AIAgent(config)
            st.session_state.last_config = vars(config)
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

def main():
    """
    Main application function
    
    Handles the overall flow of the application, including:
    - Setting up the sidebar
    - Validating API keys
    - Initializing session state
    - Processing user queries
    - Displaying results and token usage
    """
    st.title("AI Agent Assistant")
    
    # Setup sidebar and get configuration
    config = setup_sidebar()
    
    # Only proceed if all keys are valid
    if not config.validate_keys():
        st.warning("Please provide all required API keys in the sidebar.")
        return
    
    # Initialize session state
    init_session_state(config)
    
    # Query input
    query = st.text_input("Enter your query", key="query_input")
    
    # Process query
    if query:
        with st.spinner("Processing your query..."):
            response = st.session_state.agent.process_query(query)
            
            if response.get("success", False):
                st.write("Response:", response["output"])
                
                # Display token usage
                token_usage = response.get("token_usage", {})
                with st.expander("Token Usage"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Prompt Tokens", token_usage.get("prompt_tokens", 0))
                    col2.metric("Completion Tokens", token_usage.get("completion_tokens", 0))
                    col3.metric("Total Tokens", token_usage.get("total_tokens", 0))
                    col4.metric("Cost", token_usage.get("total_cost", "$0.00"))
                
                # Display intermediate steps in an expander
                with st.expander("View thinking process"):
                    for step in response.get("intermediate_steps", []):
                        st.write(step)
            else:
                st.error(f"Error: {response.get('error', 'Unknown error occurred')}")

if __name__ == "__main__":
    main()
