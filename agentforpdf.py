from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os

# Initialize the ChatGoogleGenerativeAI LLM with your API key and Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Specify the Gemini model version
    temperature=0.5,         # Control creativity (lower is more deterministic)
    max_tokens=None,         # Set maximum token length (None means no limit)
    timeout=None,            # Set timeout (None means no timeout)
    max_retries=2,           # Number of retries if the API fails
    api_key="YOUR_GOOGLE_API_KEY"  # Provide your Google API key here
)

# Load the PDF document (replace with your PDF's path)
document_path = 'your_pdf_document.pdf'  # Path to your PDF document
loader = PyPDFLoader(document_path)
documents = loader.load()

# Combine the document content into a single text string
document_content = "\n".join([doc.page_content for doc in documents])

# Set up a ConversationChain to handle the chatbot's dialogue
conversation_chain = ConversationChain(
    llm=llm, 
    prompt=PromptTemplate(input_variables=["input"], template="{input}")
)

# Function to interact with the chatbot (AI-powered)
def chat_with_pdf():
    print("Welcome to the PDF-based AI chatbot! Ask me anything about the PDF.")
    print("Type 'exit' to quit the chatbot.")
    
    # User prompt for questions
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # AI-powered prompt to analyze the user's input and provide relevant responses
        ai_prompt = f"""
        You are an AI chatbot that can answer questions based on the content of a PDF document. 
        You are provided with the following document text:
        
        Document Content: 
        {document_content}

        The user asks: "{user_input}"

        Please answer the user's question based on the document content.
        """
        
        # Generate the chatbot's response
        response = llm.generate([ai_prompt])
        
        # Print the response
        print(f"Bot: {response[0].text}")

# Run the chatbot
chat_with_pdf()
