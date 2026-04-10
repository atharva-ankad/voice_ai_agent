from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the tiny local LLM
# temperature=0 makes it deterministic and less creative (good for classification)
print("Initializing Llama 3.2:1B for Intent Detection...")
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# We design a strict prompt to ensure the LLM doesn't chat, but only classifies.
intent_prompt = PromptTemplate(
    input_variables=["text"],
    template="""You are a strict intent classifier.

    Your job is to classify the user's request into EXACTLY ONE category:

    create_file
    write_code
    summarize_text
    general_chat

    Definitions:

    create_file
    User asks to create an empty file or folder. No code generation.

    write_code
    User asks to generate or write code, a script, a function, or program.
    Examples: python file, script, function, code snippet.

    summarize_text
    User asks to summarize, shorten, or provide a summary of text.

    general_chat
    Anything else: questions, explanations, greetings, or conversation.

    Rules:

    Respond with ONLY the category name.
    Do not explain your answer.
    Do not add punctuation or extra words.
    Choose the closest matching category.

    Examples:

    User: create a new file called test.txt
    Output: create_file

    User: write a python function to retry requests
    Output: write_code

    User: summarize this paragraph about machine learning
    Output: summarize_text

    User: what is artificial intelligence
    Output: general_chat

    User Input: "{text}"
    Intent:"""
)

# Connect the prompt, model, and a string parser into a LangChain pipeline
intent_chain = intent_prompt | llm | StrOutputParser()

def detect_intent(text):
    """
    Takes transcribed text and returns one of the 4 supported intents.
    """
    if not text or "Error" in text:
        return "error_no_text"
        
    try:
        # Ask the LLM
        raw_intent = intent_chain.invoke({"text": text}).strip().lower()
        
        # Safety fallback: even with temperature=0, small LLMs sometimes add extra words.
        # We enforce strict routing.
        valid_intents = ["create_file", "write_code", "summarize_text", "general_chat"]
        for v_intent in valid_intents:
            if v_intent in raw_intent:
                return v_intent
                
        return "general_chat"  # Default fallback if confused
    except Exception as e:
        return f"error: {str(e)}"