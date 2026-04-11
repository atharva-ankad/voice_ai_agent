import os
import re
from langchain_core.prompts import PromptTemplate # pyright: ignore[reportMissingImports]
from langchain_core.output_parsers import StrOutputParser # pyright: ignore[reportMissingImports]

# SAFETY CONSTRAINT: Lock all operations to output/ folder
OUTPUT_DIR = os.path.abspath("output")
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


def _safe_filepath(filename: str) -> str | None:
    """
    Resolves the final path and hard-rejects anything that escapes OUTPUT_DIR.
    Returns None if the path is unsafe.
    """
    # Strip any directory components the LLM might hallucinate (e.g. ../../evil.py)
    basename = os.path.basename(filename.strip())
    if not basename:
        return None

    resolved = os.path.realpath(os.path.join(OUTPUT_DIR, basename))

    # Hard boundary: resolved path must start with OUTPUT_DIR
    if not resolved.startswith(OUTPUT_DIR + os.sep):
        return None

    return resolved


def _extract_metadata(text: str, llm) -> tuple[str, str]:
    """Single LLM call to get filename and write mode."""
    prompt = PromptTemplate.from_template(
        "Request: {text}\n"
        "Reply with ONLY: filename,mode\n"
        "mode is 'a' to append to existing file, 'w' to create/overwrite.\n"
        "Example: app.py,w\n"
        "Output:"
    )
    try:
        raw = (prompt | llm | StrOutputParser()).invoke({"text": text}).strip()
        filename, mode = raw.splitlines()[0].split(",")
        return filename.strip(), mode.strip().lower()
    except Exception:
        return "code_snippet.py", "w"


def _generate_code(text: str, llm, context: str = "") -> str:
    """Single LLM call to generate code, optionally aware of existing file content."""
    if context:
        prompt = PromptTemplate.from_template(
            "Existing code:\n{context}\n\n"
            "Request: {text}\n"
            "Write only the new code to append. No markdown, no explanation.\n"
            "Code:"
        )
        raw = (prompt | llm | StrOutputParser()).invoke({"text": text, "context": context})
    else:
        prompt = PromptTemplate.from_template(
            "Request: {text}\n"
            "Write only the code. No markdown, no explanation.\n"
            "Code:"
        )
        raw = (prompt | llm | StrOutputParser()).invoke({"text": text})

    clean = re.sub(r"^```[\w]*\n?", "", raw.strip())
    clean = re.sub(r"\n?```$", "", clean).strip()
    return clean


def write_code_tool(text: str, llm) -> dict:
    """
    Intent: write_code
    Generates code and writes or appends it to a file.
    ALL writes are strictly sandboxed to the output/ folder.
    """
    # Step 1: Get filename and mode from LLM
    raw_filename, raw_mode = _extract_metadata(text, llm)

    # Step 2: Sanitize and enforce output/ boundary
    safe_name = sanitize_filename(raw_filename) or "code_snippet.py"
    filepath = _safe_filepath(safe_name)
    if not filepath:
        return {
            "action": "write_code blocked",
            "result": f"Unsafe file path rejected: '{raw_filename}'. All writes must stay inside output/."
        }

    file_mode = "a" if raw_mode == "a" else "w"
    file_exists = os.path.exists(filepath)
    safe_name = os.path.basename(filepath)

    # Step 3: Load tail of existing file for append context (capped for Phi3)
    context = ""
    if file_mode == "a" and file_exists:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                context = f.read()[-800:]
        except IOError:
            pass

    # Step 4: Generate code
    code = _generate_code(text, llm, context=context)
    if not code:
        return {"action": "write_code failed", "result": "Model returned empty output."}

    # Step 5: Write — output/ is guaranteed to exist (created in app.py), but makedirs as safety net
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(filepath, file_mode, encoding="utf-8") as f:
            if file_mode == "a" and file_exists:
                f.write("\n\n# --- Appended ---\n\n")
            f.write(code)

        action = "Appended to" if (file_mode == "a" and file_exists) else "Created"
        return {
            "action": f"{action} {safe_name}",
            "result": f"Code written to output/{safe_name}"
        }
    except IOError as e:
        return {"action": "write_code failed", "result": f"I/O error: {e}"}
    
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