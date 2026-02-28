import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Loads your GEMINI_API_KEY from the .env file

app = FastAPI()

# CORS lets browsers/testers talk to your API — required for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# What the user sends IN
class CodeRequest(BaseModel):
    code: str

# What you send back OUT
class CodeResponse(BaseModel):
    error: List[int]   # Line numbers with errors (empty list if none)
    result: str        # The actual output or traceback text

# What the AI returns (internal use)
class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def execute_python_code(code: str) -> dict:
    """Runs Python code, captures output. Returns success status + output text."""
    
    # Redirect stdout so we can capture print() statements
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)  # Actually runs the code string
        output = sys.stdout.getvalue().strip()  # Grab everything that was printed
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()  # Grab the full error message
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout  # Always restore stdout, even if something broke

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """Sends code + error to Gemini AI. Gets back a list of broken line numbers."""
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return the line number(s) where the error is located.
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    # Parse the AI's JSON response into our Pydantic model
    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    
    # Step 1: Run the code
    execution_result = execute_python_code(request.code)

    # Step 2: Did it succeed?
    if execution_result["success"]:
        # No errors — return output with empty error list
        return CodeResponse(
            error=[],
            result=execution_result["output"]
        )
    else:
        # Step 3: There was an error — ask AI to analyze it
        error_lines = analyze_error_with_ai(
            code=request.code,
            traceback_text=execution_result["output"]
        )
        
        # Step 4: Return the traceback + which lines the AI flagged
        return CodeResponse(
            error=error_lines,
            result=execution_result["output"]
        )
