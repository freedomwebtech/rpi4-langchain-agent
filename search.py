from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the ChatGoogleGenerativeAI LLM with your API key and Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Specify the Gemini model version
    temperature=0.5,           # Control creativity (lower is more deterministic)
    max_tokens=None,         # Set maximum token length (None means no limit)
    timeout=None,            # Set timeout (None means no timeout)
    max_retries=2,           # Number of retries if the API fails
    api_key=""  # Provide your Google API key here
)

# Initialize DuckDuckGo search tool for web search capability
search_tool = DuckDuckGoSearchRun()

# Define a list of tools (e.g., DuckDuckGo search tool)
tools = [
    Tool(
        name="DuckDuckGoSearch",
        func=search_tool.run,
        description="Search the web using DuckDuckGo"
    )
]

# Initialize the agent with the LLM and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="ZERO_SHOT_REACT_DESCRIPTION",  # Specify the agent type (reacts to description)
    verbose=True  # Enable verbose output for better debugging
)

# Define a prompt for the agent to respond to
prompt = "tell me about huggingface"

# Run the agent to get a response, which may include search results
response = agent.run(prompt)

# Print the response from the agent
print(response)
