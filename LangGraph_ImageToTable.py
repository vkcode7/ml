"""
PDF Financial Statement Table Extractor using OpenAI GPT-4o and LangGraph
Extracts tables from PDF pages with retry logic (max 3 attempts)
"""

import pandas as pd
import base64
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import json


class TableExtractionState(TypedDict):
    """State definition for the LangGraph workflow"""
    pdf_page_image: str  # base64 encoded image
    attempt: int
    table_data: dict | None
    dataframe: pd.DataFrame | None
    error: str | None
    is_valid_table: bool


class PDFTableExtractor:
    """
    A class to extract financial statement tables from PDF pages using 
    OpenAI GPT-4o and LangGraph with retry logic (max 3 attempts)
    """
    
    def __init__(self, api_key: str, max_attempts: int = 3):
        """
        Initialize the PDF Table Extractor
        
        Args:
            api_key: OpenAI API key
            max_attempts: Maximum number of extraction attempts (default: 3)
        """
        self.client = OpenAI(api_key=api_key)
        self.max_attempts = max_attempts
        self.graph = None
        
    @staticmethod
    def encode_pdf_page_to_base64(file_path: str) -> str:
        """
        Convert an image file to base64 encoded string.
        
        Args:
            file_path: Path to the image file (JPEG, PNG, etc.)
            
        Returns:
            Base64 encoded string of the image
            
        Note:
            For PDF files, use pdf2image library first:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            # Save image to temp file, then use this method
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _extract_table_with_gpt4o(self, state: TableExtractionState) -> TableExtractionState:
        """
        Extract table from PDF page using GPT-4o vision
        
        Args:
            state: Current state of the extraction process
            
        Returns:
            Updated state with extracted table data or error
        """
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
            
            response = self.client.chat.completions.create(
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
    
    def _convert_to_dataframe(self, state: TableExtractionState) -> TableExtractionState:
        """
        Convert extracted table data to pandas DataFrame
        
        Args:
            state: Current state with extracted table data
            
        Returns:
            Updated state with DataFrame or validation error
        """
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
    
    def _should_retry(self, state: TableExtractionState) -> str:
        """
        Decide whether to retry extraction or end the process
        
        Args:
            state: Current state of extraction
            
        Returns:
            String indicating next action: "end", "retry", or "max_attempts"
        """
        # If valid table, we're done
        if state["is_valid_table"] and state["dataframe"] is not None:
            return "end"
        
        # If no table was found in PDF (has_table: false), we're done
        if (state["table_data"] and 
            not state["table_data"].get("has_table") and 
            state["dataframe"] is not None):
            return "end"
        
        # If we've reached max attempts, return empty dataframe
        if state["attempt"] >= self.max_attempts:
            print(f"Max attempts ({self.max_attempts}) reached. Returning empty DataFrame.")
            return "max_attempts"
        
        # Otherwise, retry
        print(f"Retrying... (attempt {state['attempt']} of {self.max_attempts})")
        return "retry"
    
    def _return_empty_dataframe(self, state: TableExtractionState) -> TableExtractionState:
        """
        Return empty DataFrame after max attempts reached
        
        Args:
            state: Current state
            
        Returns:
            State with empty DataFrame
        """
        return {
            **state,
            "dataframe": pd.DataFrame(),
            "is_valid_table": True,
            "error": f"Max attempts ({self.max_attempts}) reached - returning empty DataFrame"
        }
    
    def _create_extraction_graph(self) -> StateGraph:
        """
        Create the LangGraph workflow for table extraction
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(TableExtractionState)
        
        # Add nodes
        workflow.add_node("extract", self._extract_table_with_gpt4o)
        workflow.add_node("convert", self._convert_to_dataframe)
        workflow.add_node("empty_result", self._return_empty_dataframe)
        
        # Set entry point
        workflow.set_entry_point("extract")
        
        # Add edges
        workflow.add_edge("extract", "convert")
        
        # Add conditional edges from convert
        workflow.add_conditional_edges(
            "convert",
            self._should_retry,
            {
                "end": END,
                "retry": "extract",
                "max_attempts": "empty_result"
            }
        )
        
        # Empty result goes to end
        workflow.add_edge("empty_result", END)
        
        return workflow.compile()
    
    def extract_table(self, pdf_page_image_base64: str) -> pd.DataFrame:
        """
        Extract financial table from PDF page (main public method)
        
        Args:
            pdf_page_image_base64: Base64 encoded PDF page image
            
        Returns:
            pd.DataFrame: Extracted table or empty DataFrame if:
                - No table found in PDF
                - Extraction failed after max attempts
                - Invalid table structure
        
        Example:
            >>> extractor = PDFTableExtractor(api_key="sk-...")
            >>> image_base64 = PDFTableExtractor.encode_pdf_page_to_base64("page.jpg")
            >>> df = extractor.extract_table(image_base64)
            >>> print(df.head())
        """
        # Create the graph if not already created
        if self.graph is None:
            self.graph = self._create_extraction_graph()
        
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
        final_state = self.graph.invoke(initial_state)
        
        # Return the DataFrame
        return final_state["dataframe"]
    
    def extract_financial_table_from_pdf(self, pdf_page_image_base64: str) -> pd.DataFrame:
        """
        Alias for extract_table method for backward compatibility
        
        Args:
            pdf_page_image_base64: Base64 encoded PDF page image
            
        Returns:
            pd.DataFrame: Extracted table or empty DataFrame
        """
        return self.extract_table(pdf_page_image_base64)


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    extractor = PDFTableExtractor(api_key="your-api-key-here", max_attempts=3)
    
    # Example 1: Using with an image file
    # pdf_page_base64 = PDFTableExtractor.encode_pdf_page_to_base64("path/to/pdf_page.jpg")
    # df = extractor.extract_table(pdf_page_base64)
    # print(df)
    
    # Example 2: Using with pdf2image
    # from pdf2image import convert_from_path
    # import io
    # 
    # images = convert_from_path("financial_statement.pdf", first_page=1, last_page=1)
    # image_bytes = io.BytesIO()
    # images[0].save(image_bytes, format='JPEG')
    # pdf_page_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    # 
    # df = extractor.extract_table(pdf_page_base64)
    # print(df)
    
    # Example 3: Process multiple pages
    # for page_num in range(1, 6):
    #     images = convert_from_path("report.pdf", first_page=page_num, last_page=page_num)
    #     image_bytes = io.BytesIO()
    #     images[0].save(image_bytes, format='JPEG')
    #     page_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    #     
    #     df = extractor.extract_table(page_base64)
    #     if not df.empty:
    #         print(f"Page {page_num}: Found table with {len(df)} rows")
    #         print(df.head())
    
    print("PDF Financial Table Extractor ready!")
    print("\nUsage:")
    print("  extractor = PDFTableExtractor(api_key='your-key', max_attempts=3)")
    print("  image_base64 = PDFTableExtractor.encode_pdf_page_to_base64('page.jpg')")
    print("  df = extractor.extract_table(image_base64)")
    print("\nOr use the alias method:")
    print("  df = extractor.extract_financial_table_from_pdf(image_base64)
