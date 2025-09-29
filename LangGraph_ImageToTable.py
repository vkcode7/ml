"""
PDF Financial Statement Table Extractor using OpenAI GPT-4o and LangGraph
Extracts tables from PDF pages with retry logic (max 3 attempts)
"""

import pandas as pd
import base64
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from openai import OpenAI
import json
import io

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")

# Define the state for our graph
class TableExtractionState(TypedDict):
    pdf_page_image: str  # base64 encoded image or path
    attempt: int
    table_data: dict | None
    dataframe: pd.DataFrame | None
    error: str | None
    is_valid_table: bool

def encode_pdf_page_to_base64(pdf_path: str, page_num: int = 0) -> str:
    """
    Convert a PDF page to base64 encoded image.
    You'll need to use pdf2image or similar library for this.
    """
    # For demonstration - assumes you have the image already
    # In practice, use: from pdf2image import convert_from_path
    # images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num+1)
    # return base64.b64encode(images[0].tobytes()).decode('utf-8')
    
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_table_with_gpt4o(state: TableExtractionState) -> TableExtractionState:
    """Extract table from PDF page using GPT-4o vision"""
    
    attempt = state["attempt"] + 1
    print(f"Attempt {attempt}: Extracting table from PDF...")
    
    try:
        # Create the prompt based on attempt number
        if attempt == 1:
            prompt = """Analyze this PDF page and extract any financial statement table you find.
            
            Return the data in JSON format with the following structure:
            {
                "has_table": true/false,
                "table": {
                    "headers": ["column1", "column2", ...],
                    "rows": [
                        ["value1", "value2", ...],
                        ["value1", "value2", ...]
                    ]
                }
            }
            
            If there's no table, set "has_table" to false and "table" to null.
            Ensure all rows have the same number of columns as headers.
            Preserve numerical values exactly as they appear."""
        else:
            prompt = f"""This is attempt {attempt} to extract a financial table from this PDF page.
            Previous attempt had issues with the table structure.
            
            Please carefully extract the table ensuring:
            1. All rows have exactly the same number of columns as headers
            2. Numerical values are preserved accurately
            3. Empty cells are represented as empty strings ""
            4. Column headers are clear and descriptive
            
            Return in this exact JSON format:
            {{
                "has_table": true/false,
                "table": {{
                    "headers": ["column1", "column2", ...],
                    "rows": [
                        ["value1", "value2", ...],
                        ["value1", "value2", ...]
                    ]
                }}
            }}"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{state['pdf_page_image']}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        table_data = json.loads(content)
        
        return {
            **state,
            "attempt": attempt,
            "table_data": table_data,
            "error": None
        }
        
    except Exception as e:
        print(f"Error in extraction attempt {attempt}: {str(e)}")
        return {
            **state,
            "attempt": attempt,
            "error": str(e),
            "table_data": None
        }

def convert_to_dataframe(state: TableExtractionState) -> TableExtractionState:
    """Convert extracted table data to pandas DataFrame"""
    
    print("Converting table data to DataFrame...")
    
    try:
        table_data = state["table_data"]
        
        # Check if table exists
        if not table_data or not table_data.get("has_table"):
            print("No table found in the PDF page")
            return {
                **state,
                "dataframe": pd.DataFrame(),
                "is_valid_table": True,  # Empty is valid
                "error": None
            }
        
        # Extract headers and rows
        table = table_data.get("table", {})
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        if not headers or not rows:
            print("Table structure is incomplete")
            return {
                **state,
                "dataframe": pd.DataFrame(),
                "is_valid_table": False,
                "error": "Incomplete table structure"
            }
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Validate: check if all rows have correct number of columns
        if df.shape[1] != len(headers):
            raise ValueError("Column count mismatch")
        
        # Check for completely empty dataframe
        if df.empty or (df.shape[0] == 0):
            raise ValueError("DataFrame is empty")
        
        print(f"Successfully created DataFrame with shape {df.shape}")
        
        return {
            **state,
            "dataframe": df,
            "is_valid_table": True,
            "error": None
        }
        
    except Exception as e:
        print(f"Error converting to DataFrame: {str(e)}")
        return {
            **state,
            "dataframe": None,
            "is_valid_table": False,
            "error": str(e)
        }

def should_retry(state: TableExtractionState) -> str:
    """Decide whether to retry extraction or end"""
    
    # If valid table, we're done
    if state["is_valid_table"] and state["dataframe"] is not None:
        return "end"
    
    # If no table was found in PDF (has_table: false), we're done
    if (state["table_data"] and 
        not state["table_data"].get("has_table") and 
        state["dataframe"] is not None):
        return "end"
    
    # If we've reached max attempts, return empty dataframe
    if state["attempt"] >= 3:
        print("Max attempts reached. Returning empty DataFrame.")
        return "max_attempts"
    
    # Otherwise, retry
    print(f"Retrying... (attempt {state['attempt']} of 3)")
    return "retry"

def return_empty_dataframe(state: TableExtractionState) -> TableExtractionState:
    """Return empty DataFrame after max attempts"""
    return {
        **state,
        "dataframe": pd.DataFrame(),
        "is_valid_table": True,
        "error": "Max attempts reached - returning empty DataFrame"
    }

# Build the LangGraph
def create_table_extraction_graph():
    """Create the LangGraph workflow for table extraction"""
    
    workflow = StateGraph(TableExtractionState)
    
    # Add nodes
    workflow.add_node("extract", extract_table_with_gpt4o)
    workflow.add_node("convert", convert_to_dataframe)
    workflow.add_node("empty_result", return_empty_dataframe)
    
    # Set entry point
    workflow.set_entry_point("extract")
    
    # Add edges
    workflow.add_edge("extract", "convert")
    
    # Add conditional edges from convert
    workflow.add_conditional_edges(
        "convert",
        should_retry,
        {
            "end": END,
            "retry": "extract",
            "max_attempts": "empty_result"
        }
    )
    
    # Empty result goes to end
    workflow.add_edge("empty_result", END)
    
    return workflow.compile()

# Main function to extract table from PDF
def extract_financial_table_from_pdf(pdf_page_image_base64: str) -> pd.DataFrame:
    """
    Main function to extract financial table from PDF page
    
    Args:
        pdf_page_image_base64: Base64 encoded PDF page image
        
    Returns:
        pd.DataFrame: Extracted table or empty DataFrame
    """
    
    # Create the graph
    graph = create_table_extraction_graph()
    
    # Initialize state
    initial_state = {
        "pdf_page_image": pdf_page_image_base64,
        "attempt": 0,
        "table_data": None,
        "dataframe": None,
        "error": None,
        "is_valid_table": False
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Return the DataFrame
    return final_state["dataframe"]

# Example usage
if __name__ == "__main__":
    # Example: Load a PDF page (you'll need to implement PDF to image conversion)
    # from pdf2image import convert_from_path
    # images = convert_from_path("financial_statement.pdf", first_page=1, last_page=1)
    # image_bytes = io.BytesIO()
    # images[0].save(image_bytes, format='JPEG')
    # pdf_page_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    
    # For demonstration purposes with a sample image
    # pdf_page_base64 = encode_pdf_page_to_base64("path/to/your/pdf_page.jpg")
    
    # Extract table
    # df = extract_financial_table_from_pdf(pdf_page_base64)
    # print(df)
    
    print("PDF Financial Table Extractor ready!")
    print("Use extract_financial_table_from_pdf(base64_image) to extract tables.")
