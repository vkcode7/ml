"""
PDF Financial Statement Table Extractor for 10-K SEC Filings
Extracts Balance Sheet, Income Statement, and Cash Flow Statement from PDF files
Uses OpenAI GPT-4o and LangGraph with retry logic (max 3 attempts per page)
"""

import pandas as pd
import base64
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
from pdf2image import convert_from_path
import json
import io
from enum import Enum


class StatementType(Enum):
    """Enum for financial statement types"""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    UNKNOWN = "unknown"


class TableExtractionState(TypedDict):
    """State definition for the LangGraph workflow"""
    pdf_page_image: str  # base64 encoded image
    attempt: int
    table_data: dict | None
    dataframe: pd.DataFrame | None
    statement_type: str
    error: str | None
    is_valid_table: bool


class FinancialStatement:
    """Container for a financial statement with metadata"""
    def __init__(self, dataframe: pd.DataFrame, statement_type: StatementType, 
                 page_number: int, confidence: str = "high"):
        self.dataframe = dataframe
        self.statement_type = statement_type
        self.page_number = page_number
        self.confidence = confidence
        
    def __repr__(self):
        return (f"FinancialStatement(type={self.statement_type.value}, "
                f"page={self.page_number}, shape={self.dataframe.shape}, "
                f"confidence={self.confidence})")


class PDFFinancialStatementExtractor:
    """
    Extracts Balance Sheet, Income Statement, and Cash Flow Statement 
    from 10-K SEC filing PDFs using OpenAI GPT-4o and LangGraph
    """
    
    def __init__(self, api_key: str, max_attempts: int = 3, dpi: int = 200):
        """
        Initialize the PDF Financial Statement Extractor
        
        Args:
            api_key: OpenAI API key
            max_attempts: Maximum number of extraction attempts per page (default: 3)
            dpi: DPI for PDF to image conversion (default: 200, higher = better quality)
        """
        self.client = OpenAI(api_key=api_key)
        self.max_attempts = max_attempts
        self.dpi = dpi
        self.graph = None
        
    def convert_pdf_to_images(self, pdf_path: str, start_page: int = 1, 
                             end_page: Optional[int] = None) -> List[str]:
        """
        Convert PDF pages to base64 encoded images
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (inclusive). If None, converts all pages
            
        Returns:
            List of base64 encoded image strings
        """
        print(f"Converting PDF pages {start_page} to {end_page or 'end'}...")
        
        images = convert_from_path(
            pdf_path, 
            first_page=start_page,
            last_page=end_page,
            dpi=self.dpi
        )
        
        base64_images = []
        for img in images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            base64_images.append(img_base64)
        
        print(f"Converted {len(base64_images)} pages to images")
        return base64_images
    
    def _identify_statement_type(self, table_data: dict) -> StatementType:
        """
        Identify the type of financial statement from the extracted table
        
        Args:
            table_data: Dictionary containing extracted table and metadata
            
        Returns:
            StatementType enum value
        """
        if not table_data or not table_data.get("has_table"):
            return StatementType.UNKNOWN
        
        statement_type = table_data.get("statement_type", "").lower()
        
        if "balance" in statement_type or "balance_sheet" in statement_type:
            return StatementType.BALANCE_SHEET
        elif "income" in statement_type or "income_statement" in statement_type or \
             "operations" in statement_type or "earnings" in statement_type:
            return StatementType.INCOME_STATEMENT
        elif "cash" in statement_type or "cash_flow" in statement_type:
            return StatementType.CASH_FLOW
        else:
            return StatementType.UNKNOWN
    
    def _extract_table_with_gpt4o(self, state: TableExtractionState) -> TableExtractionState:
        """
        Extract table from PDF page using GPT-4o vision and identify statement type
        
        Args:
            state: Current state of the extraction process
            
        Returns:
            Updated state with extracted table data or error
        """
        attempt = state["attempt"] + 1
        print(f"Attempt {attempt}: Extracting and classifying financial statement...")
        
        try:
            # Create the prompt based on attempt number
            if attempt == 1:
                prompt = """Analyze this PDF page from a 10-K SEC filing and:

1. Identify if this page contains one of these financial statements:
   - Balance Sheet (or Statement of Financial Position)
   - Income Statement (or Statement of Operations/Earnings)
   - Cash Flow Statement (or Statement of Cash Flows)

2. If it contains one of these statements, extract the table data.

Return the data in JSON format:
{
    "has_table": true/false,
    "statement_type": "balance_sheet" | "income_statement" | "cash_flow" | "unknown",
    "confidence": "high" | "medium" | "low",
    "table": {
        "headers": ["column1", "column2", ...],
        "rows": [
            ["value1", "value2", ...],
            ["value1", "value2", ...]
        ]
    }
}

Guidelines:
- If the page doesn't contain any of these 3 statements, set "has_table" to false
- Look for key indicators like "Assets", "Liabilities", "Revenue", "Net Income", "Operating Activities"
- Ensure all rows have the same number of columns as headers
- Preserve numerical values and formatting (including parentheses for negatives)
- For multi-year statements, include all year columns"""
            else:
                prompt = f"""This is attempt {attempt} to extract a financial statement from a 10-K filing.
Previous attempt had issues. Please carefully:

1. Identify the statement type (Balance Sheet, Income Statement, or Cash Flow Statement)
2. Extract the complete table with proper structure

Return in this exact JSON format:
{{
    "has_table": true/false,
    "statement_type": "balance_sheet" | "income_statement" | "cash_flow" | "unknown",
    "confidence": "high" | "medium" | "low",
    "table": {{
        "headers": ["column1", "column2", ...],
        "rows": [
            ["value1", "value2", ...],
            ["value1", "value2", ...]
        ]
    }}
}}

Ensure:
- All rows have exactly the same number of columns as headers
- Numerical values are preserved accurately (including negatives in parentheses)
- Empty cells are represented as empty strings ""
- Column headers clearly indicate the time period/year"""
            
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
            
            # Identify statement type
            statement_type = self._identify_statement_type(table_data)
            
            return {
                **state,
                "attempt": attempt,
                "table_data": table_data,
                "statement_type": statement_type.value,
                "error": None
            }
            
        except Exception as e:
            print(f"Error in extraction attempt {attempt}: {str(e)}")
            return {
                **state,
                "attempt": attempt,
                "error": str(e),
                "table_data": None,
                "statement_type": StatementType.UNKNOWN.value
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
                print("No relevant financial statement found on this page")
                return {
                    **state,
                    "dataframe": pd.DataFrame(),
                    "is_valid_table": True,  # Empty is valid (not a target statement)
                    "error": None
                }
            
            # Check if it's one of the target statement types
            statement_type = state.get("statement_type", "unknown")
            if statement_type == StatementType.UNKNOWN.value:
                print("Table found but not a target financial statement")
                return {
                    **state,
                    "dataframe": pd.DataFrame(),
                    "is_valid_table": True,  # Not a target statement
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
            
            print(f"Successfully created DataFrame with shape {df.shape} for {statement_type}")
            
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
        # If valid table (or no target statement found), we're done
        if state["is_valid_table"] and state["dataframe"] is not None:
            return "end"
        
        # If no table was found in PDF (has_table: false), we're done
        if (state["table_data"] and 
            not state["table_data"].get("has_table") and 
            state["dataframe"] is not None):
            return "end"
        
        # If we've reached max attempts, return empty dataframe
        if state["attempt"] >= self.max_attempts:
            print(f"Max attempts ({self.max_attempts}) reached. Moving to next page.")
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
            "error": f"Max attempts ({self.max_attempts}) reached - moving to next page"
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
    
    def extract_from_page(self, pdf_page_image_base64: str) -> tuple[pd.DataFrame, StatementType, str]:
        """
        Extract financial statement from a single PDF page
        
        Args:
            pdf_page_image_base64: Base64 encoded PDF page image
            
        Returns:
            Tuple of (DataFrame, StatementType, confidence)
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
            "statement_type": StatementType.UNKNOWN.value,
            "error": None,
            "is_valid_table": False
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Get statement type
        statement_type_str = final_state.get("statement_type", StatementType.UNKNOWN.value)
        statement_type = StatementType(statement_type_str)
        
        # Get confidence
        confidence = "unknown"
        if final_state.get("table_data"):
            confidence = final_state["table_data"].get("confidence", "unknown")
        
        return final_state["dataframe"], statement_type, confidence
    
    def extract_from_pdf(self, pdf_path: str, start_page: int = 1, 
                        end_page: Optional[int] = None) -> Dict[StatementType, List[FinancialStatement]]:
        """
        Extract all financial statements from a 10-K PDF file
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (1-indexed, default: 1)
            end_page: Ending page number (inclusive). If None, processes all pages
            
        Returns:
            Dictionary mapping StatementType to list of FinancialStatement objects
            
        Example:
            >>> extractor = PDFFinancialStatementExtractor(api_key="sk-...")
            >>> results = extractor.extract_from_pdf("10k_filing.pdf")
            >>> 
            >>> # Access balance sheets
            >>> for stmt in results[StatementType.BALANCE_SHEET]:
            >>>     print(f"Page {stmt.page_number}:")
            >>>     print(stmt.dataframe.head())
        """
        print(f"\n{'='*60}")
        print(f"Processing 10-K PDF: {pdf_path}")
        print(f"{'='*60}\n")
        
        # Convert PDF to images
        page_images = self.convert_pdf_to_images(pdf_path, start_page, end_page)
        
        # Initialize results dictionary
        results = {
            StatementType.BALANCE_SHEET: [],
            StatementType.INCOME_STATEMENT: [],
            StatementType.CASH_FLOW: []
        }
        
        # Process each page
        actual_start = start_page
        for i, page_image in enumerate(page_images):
            page_num = actual_start + i
            print(f"\n--- Processing Page {page_num} ---")
            
            df, statement_type, confidence = self.extract_from_page(page_image)
            
            if not df.empty and statement_type != StatementType.UNKNOWN:
                financial_stmt = FinancialStatement(df, statement_type, page_num, confidence)
                results[statement_type].append(financial_stmt)
                print(f"✓ Found {statement_type.value} on page {page_num} (confidence: {confidence})")
            else:
                print(f"✗ No target financial statement found on page {page_num}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Balance Sheets found: {len(results[StatementType.BALANCE_SHEET])}")
        print(f"Income Statements found: {len(results[StatementType.INCOME_STATEMENT])}")
        print(f"Cash Flow Statements found: {len(results[StatementType.CASH_FLOW])}")
        print(f"{'='*60}\n")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    extractor = PDFFinancialStatementExtractor(
        api_key="your-api-key-here",
        max_attempts=3,
        dpi=200  # Higher DPI = better quality but slower
    )
    
    # Example 1: Extract from entire PDF
    # results = extractor.extract_from_pdf("10k_filing.pdf")
    
    # Example 2: Extract from specific page range (e.g., pages 30-50)
    # results = extractor.extract_from_pdf("10k_filing.pdf", start_page=30, end_page=50)
    
    # Example 3: Access extracted statements
    # # Get all balance sheets
    # for stmt in results[StatementType.BALANCE_SHEET]:
    #     print(f"\nBalance Sheet from Page {stmt.page_number}:")
    #     print(f"Confidence: {stmt.confidence}")
    #     print(f"Shape: {stmt.dataframe.shape}")
    #     print(stmt.dataframe.head())
    # 
    # # Get all income statements
    # for stmt in results[StatementType.INCOME_STATEMENT]:
    #     print(f"\nIncome Statement from Page {stmt.page_number}:")
    #     print(stmt.dataframe.head())
    # 
    # # Get all cash flow statements
    # for stmt in results[StatementType.CASH_FLOW]:
    #     print(f"\nCash Flow Statement from Page {stmt.page_number}:")
    #     print(stmt.dataframe.head())
    
    # Example 4: Export to Excel
    # with pd.ExcelWriter("financial_statements.xlsx") as writer:
    #     for stmt_type, statements in results.items():
    #         for i, stmt in enumerate(statements):
    #             sheet_name = f"{stmt_type.value[:20]}_p{stmt.page_number}"
    #             stmt.dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("PDF Financial Statement Extractor for 10-K Filings ready!")
    print("\nUsage:")
    print("  extractor = PDFFinancialStatementExtractor(api_key='your-key')")
    print("  results = extractor.extract_from_pdf('10k_filing.pdf')")
    print("  ")
    print("  # Access statements by type:")
    print("  for stmt in results[StatementType.BALANCE_SHEET]:")
    print("      print(stmt.dataframe)")


def UsageExample():
    # Initialize
    extractor = PDFFinancialStatementExtractor(
        api_key="sk-...",
        max_attempts=3,
        dpi=200
    )
    
    # Extract from entire 10-K
    results = extractor.extract_from_pdf("apple_10k.pdf")
    
    # Or extract from specific pages (if you know where statements are)
    results = extractor.extract_from_pdf("apple_10k.pdf", start_page=30, end_page=50)
    
    # Access Balance Sheets
    for stmt in results[StatementType.BALANCE_SHEET]:
        print(f"Page {stmt.page_number}, Confidence: {stmt.confidence}")
        print(stmt.dataframe)
    
    # Access Income Statements
    for stmt in results[StatementType.INCOME_STATEMENT]:
        print(stmt.dataframe)
    
    # Export all to Excel
    with pd.ExcelWriter("financial_statements.xlsx") as writer:
        for stmt_type, statements in results.items():
            for stmt in statements:
                sheet_name = f"{stmt_type.value}_page_{stmt.page_number}"
                stmt.dataframe.to_excel(writer, sheet_name=sheet_name)

"""
{
    StatementType.BALANCE_SHEET: [FinancialStatement(...), ...],
    StatementType.INCOME_STATEMENT: [FinancialStatement(...), ...],
    StatementType.CASH_FLOW: [FinancialStatement(...), ...]
}
"""
