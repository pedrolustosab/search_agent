# AI Agent Assistant

AI Agent Assistant is a Streamlit-based web application that leverages OpenAI's language models and external search tools to provide intelligent responses to user queries.

## Features

- Interactive web interface powered by Streamlit
- Integration with OpenAI's GPT models (GPT-3.5-turbo and GPT-4)
- Support for multiple search providers (SerpAPI and Tavily)
- Dynamic configuration of API keys and model settings
- Display of token usage and costs
- Visualization of the AI's thinking process

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- OpenAI API key
- SerpAPI or Tavily API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-agent-assistant.git
   cd ai-agent-assistant
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Obtain an OpenAI API key from [OpenAI](https://openai.com/)
   - Get a SerpAPI key from [SerpAPI](https://serpapi.com/) or a Tavily API key from [Tavily](https://tavily.com/)

## Usage

1. Run the Streamlit app:
   ```
   streamlit run chat.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

3. In the sidebar:
   - Enter your OpenAI API key
   - Select your preferred search provider (SerpAPI or Tavily)
   - Enter the corresponding search provider API key
   - Optionally, adjust the model settings

4. Enter your query in the main text input field and press Enter

5. View the AI's response, token usage, and thinking process

## Configuration

You can customize the following settings in the sidebar:
- OpenAI API Key
- Search Provider (SerpAPI or Tavily)
- Search Provider API Key
- AI Model (GPT-3.5-turbo or GPT-4)
- Maximum Response Tokens

