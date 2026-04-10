import os
import re
from langchain_core.prompts import PromptTemplate # pyright: ignore[reportMissingImports]
from langchain_core.output_parsers import StrOutputParser # pyright: ignore[reportMissingImports]

# SAFETY CONSTRAINT: Lock all operations to output/ folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(llm_output: str, default_name="generated_file.txt") -> str:
    """
    Smart filename extractor. Handles chatty LLMs that output paragraphs
    instead of just the filename to prevent OS path length crashes.
    """
    # 1. Search the text for anything that looks like a filename (e.g., script.py, notes.txt)
    match = re.search(r'([a-zA-Z0-9_\-]+\.(?:py|txt|html|md|js|csv))', llm_output)
    
    if match:
        return match.group(1) # Return just the matched filename

    # 2. If no extension was found, clean the string of illegal characters
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '', llm_output.strip())
    
    # 3. If the string is ridiculously long (LLM hallucinated) or empty, use the default
    if not clean or len(clean) > 40:
        return default_name
        
    return clean

def create_file_tool(text: str, llm) -> dict:
    """Extracts a filename from the transcription and creates an empty file."""
    prompt = PromptTemplate.from_template(
        "Extract ONLY the filename mentioned in this request. If no filename is given, output 'new_file.txt'. Request: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    raw_filename = chain.invoke({"text": text})
    
    safe_name = sanitize_filename(raw_filename, "new_file.txt")
    filepath = os.path.join(OUTPUT_DIR, safe_name)
    
    try:
        with open(filepath, 'w') as f:
            f.write("")
        return {
            "action": f"Created file: {safe_name}",
            "result": f"Success. File saved to {filepath}"
        }
    except Exception as e:
        return {"action": "Failed to create file", "result": str(e)}

def write_code_tool(text: str, llm) -> dict:
    """Generates code based on the transcription and saves it to a file."""
    # 1. Generate the code
    code_prompt = PromptTemplate.from_template(
        "Write the code requested here. Output ONLY raw code. No markdown backticks, no explanations. Request: {text}"
    )
    code_chain = code_prompt | llm | StrOutputParser()
    
    # Strip backticks just in case the model ignores instructions
    generated_code = code_chain.invoke({"text": text}).replace("```python", "").replace("```", "").strip()
    
    # 2. Determine a filename
    name_prompt = PromptTemplate.from_template(
        "What should the filename be for this coding request? Output ONLY the filename ending in .py,.java, .html, etc. Request: {text}"
    )
    name_chain = name_prompt | llm | StrOutputParser()
    raw_filename = name_chain.invoke({"text": text})
    
    # Use our new smart extractor
    safe_name = sanitize_filename(raw_filename, "script.py")
    filepath = os.path.join(OUTPUT_DIR, safe_name)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        return {
            "action": f"Generated code and saved to {safe_name}",
            "result": f"Success. Code written to {filepath}"
        }
    except Exception as e:
        return {"action": "Failed to write code", "result": str(e)}

def summarize_text_tool(text: str, llm) -> dict:
    """Summarizes the provided text."""
    prompt = PromptTemplate.from_template(
        "Provide a concise summary of the following request/text: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": text})
    
    return {
        "action": "Summarized text",
        "result": summary
    }

def general_chat_tool(text: str, llm) -> dict:
    """Handles general conversational requests."""
    prompt = PromptTemplate.from_template("{text}")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": text})
    
    return {
        "action": "General Chat Response",
        "result": response
    }

def execute_action(intent: str, transcription: str, llm) -> dict:
    """Master router to trigger the correct tool based on intent."""
    if intent == "create_file":
        return create_file_tool(transcription, llm)
    elif intent == "write_code":
        return write_code_tool(transcription, llm)
    elif intent == "summarize_text":
        return summarize_text_tool(transcription, llm)
    elif intent == "general_chat":
        return general_chat_tool(transcription, llm)
    else:
        return {
            "action": "Unknown Intent",
            "result": f"System could not map intent '{intent}' to a tool."
        }