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
    statement_type: str
    has_target_statement: bool
    original_headers: List[str] | None
    normalized_headers: List[str] | None
    table_data: dict | None
    dataframe: pd.DataFrame | None
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
    
    def _identify_statement_type(self, statement_name: str) -> StatementType:
        """
        Identify the type of financial statement from the statement name
        
        Args:
            statement_name: Name of the statement extracted from the page
            
        Returns:
            StatementType enum value
        """
        statement_name_lower = statement_name.lower()
        
        if "balance" in statement_name_lower or "financial position" in statement_name_lower:
            return StatementType.BALANCE_SHEET
        elif ("income" in statement_name_lower or "operations" in statement_name_lower or 
              "earnings" in statement_name_lower or "comprehensive income" in statement_name_lower):
            return StatementType.INCOME_STATEMENT
        elif "cash" in statement_name_lower and "flow" in statement_name_lower:
            return StatementType.CASH_FLOW
        else:
            return StatementType.UNKNOWN
    
    def _extract_statement_type_and_headers_info(self, state: TableExtractionState) -> TableExtractionState:
        """
        First node: Analyze the page to identify statement type and extract header information
        
        Args:
            state: Current state of the extraction process
            
        Returns:
            Updated state with statement type and headers info
        """
        attempt = state["attempt"] + 1
        print(f"\n[Node 1] Analyzing page for statement type and headers (Attempt {attempt})...")
        
        try:
            prompt = """Analyze this PDF page from a 10-K SEC filing and:

1. Check if this page contains a financial statement table
2. If yes, identify the statement type and extract the column headers

Return ONLY this JSON structure:
{
    "has_financial_table": true/false,
    "statement_name": "exact title of the statement from the page",
    "headers": ["header1", "header2", "header3", ...],
    "confidence": "high" | "medium" | "low"
}

Statement types to look for:
- Balance Sheet (or Statement of Financial Position, Consolidated Balance Sheet)
- Income Statement (or Statement of Operations, Statement of Earnings, Statement of Comprehensive Income)
- Cash Flow Statement (or Statement of Cash Flows)

If the page doesn't contain any of these statements, set "has_financial_table" to false.
Extract ALL column headers exactly as they appear, including year/period information."""

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
                max_tokens=2048,
                temperature=0
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Check if it has a financial table
            has_table = result.get("has_financial_table", False)
            
            if not has_table:
                print("   ✗ No financial statement found on this page")
                return {
                    **state,
                    "attempt": attempt,
                    "has_target_statement": False,
                    "statement_type": StatementType.UNKNOWN.value,
                    "original_headers": None,
                    "error": None
                }
            
            # Identify statement type
            statement_name = result.get("statement_name", "")
            statement_type = self._identify_statement_type(statement_name)
            
            if statement_type == StatementType.UNKNOWN:
                print(f"   ✗ Found table but not a target statement: {statement_name}")
                return {
                    **state,
                    "attempt": attempt,
                    "has_target_statement": False,
                    "statement_type": StatementType.UNKNOWN.value,
                    "original_headers": None,
                    "error": None
                }
            
            # Extract headers
            headers = result.get("headers", [])
            confidence = result.get("confidence", "unknown")
            
            print(f"   ✓ Found {statement_type.value}: {statement_name}")
            print(f"   ✓ Extracted {len(headers)} column headers (Confidence: {confidence})")
            
            return {
                **state,
                "attempt": attempt,
                "has_target_statement": True,
                "statement_type": statement_type.value,
                "original_headers": headers,
                "error": None
            }
            
        except Exception as e:
            print(f"   ✗ Error in statement type extraction: {str(e)}")
            return {
                **state,
                "attempt": attempt,
                "has_target_statement": False,
                "statement_type": StatementType.UNKNOWN.value,
                "original_headers": None,
                "error": str(e)
            }
    
    def _normalize_column_headers(self, state: TableExtractionState) -> TableExtractionState:
        """
        Second node: Normalize column headers to "Month Year - Month Year" format
        
        Args:
            state: Current state with original headers
            
        Returns:
            Updated state with normalized headers
        """
        print(f"\n[Node 2] Normalizing column headers...")
        
        try:
            original_headers = state.get("original_headers", [])
            
            if not original_headers:
                print("   ✗ No headers to normalize")
                return {
                    **state,
                    "normalized_headers": [],
                    "error": "No headers found"
                }
            
            prompt = f"""Given these column headers from a financial statement, normalize them to a consistent format.

Original headers: {json.dumps(original_headers)}

For each header, convert it to this format:
- If it's a time period: "Month Year - Month Year" (e.g., "January 2024 - March 2024")
- If it's a single point in time: "Month Year" (e.g., "December 2024")
- If it's a full year: "Year" (e.g., "2024")
- If it's a description column (like "Assets", "Line Item", etc.): keep as is

Return ONLY a JSON array with normalized headers:
{{
    "normalized_headers": ["header1", "header2", ...]
}}

Examples:
- "Three Months Ended March 31, 2024" → "January 2024 - March 2024"
- "Year Ended December 31, 2024" → "2024"
- "As of December 31, 2024" → "December 2024"
- "As of March 31, 2024" → "March 2024"
- "2024" → "2024"
- "Assets" → "Assets"
- "Description" → "Description"

Maintain the same number and order of headers."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2048,
                temperature=0
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            normalized_headers = result.get("normalized_headers", [])
            
            print(f"   ✓ Normalized {len(normalized_headers)} headers")
            print(f"   Original: {original_headers[:3]}...")
            print(f"   Normalized: {normalized_headers[:3]}...")
            
            return {
                **state,
                "normalized_headers": normalized_headers,
                "error": None
            }
            
        except Exception as e:
            print(f"   ✗ Error normalizing headers: {str(e)}")
            # Fallback: use original headers
            return {
                **state,
                "normalized_headers": state.get("original_headers", []),
                "error": f"Header normalization failed: {str(e)}"
            }
    
    def _extract_table_with_gpt4o(self, state: TableExtractionState) -> TableExtractionState:
        """
        Third node: Extract the complete table data using the normalized headers
        
        Args:
            state: Current state with normalized headers
            
        Returns:
            Updated state with extracted table data
        """
        attempt = state["attempt"]
        print(f"\n[Node 3] Extracting table data with normalized headers...")
        
        try:
            normalized_headers = state.get("normalized_headers", [])
            statement_type = state.get("statement_type", "unknown")
            
            if not normalized_headers:
                raise ValueError("No normalized headers available for extraction")
            
            prompt = f"""Extract the complete financial statement table from this PDF page.

Statement Type: {statement_type}
Expected Column Headers (use these exactly): {json.dumps(normalized_headers)}

Return the data in this JSON format:
{{
    "has_table": true,
    "table": {{
        "headers": {json.dumps(normalized_headers)},
        "rows": [
            ["value1", "value2", ...],
            ["value1", "value2", ...]
        ]
    }},
    "confidence": "high" | "medium" | "low"
}}

Important:
- Use the exact headers provided above
- Each row must have exactly {len(normalized_headers)} values (matching the number of headers)
- Preserve numerical values exactly, including negatives in parentheses like (1,234)
- Empty cells should be represented as empty strings ""
- Extract ALL rows from the table
- Maintain proper alignment between row values and column headers"""

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
            
            print(f"   ✓ Extracted table with {len(table_data.get('table', {}).get('rows', []))} rows")
            
            return {
                **state,
                "table_data": table_data,
                "error": None
            }
            
        except Exception as e:
            print(f"   ✗ Error extracting table: {str(e)}")
            return {
                **state,
                "table_data": None,
                "error": str(e)
            }
    
    def _convert_to_dataframe(self, state: TableExtractionState) -> TableExtractionState:
        """
        Fourth node: Convert extracted table data to pandas DataFrame
        
        Args:
            state: Current state with extracted table data
            
        Returns:
            Updated state with DataFrame or validation error
        """
        print(f"\n[Node 4] Converting to DataFrame...")
        
        try:
            # If no target statement was found, return empty DataFrame
            if not state.get("has_target_statement", False):
                return {
                    **state,
                    "dataframe": pd.DataFrame(),
                    "is_valid_table": True,
                    "error": None
                }
            
            table_data = state.get("table_data")
            
            if not table_data or not table_data.get("has_table"):
                print("   ✗ No table data available")
                return {
                    **state,
                    "dataframe": pd.DataFrame(),
                    "is_valid_table": False,
                    "error": "No table data extracted"
                }
            
            # Extract headers and rows
            table = table_data.get("table", {})
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if not headers or not rows:
                print("   ✗ Incomplete table structure")
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
            
            statement_type = state.get("statement_type", "unknown")
            print(f"   ✓ Created DataFrame: {df.shape} for {statement_type}")
            
            return {
                **state,
                "dataframe": df,
                "is_valid_table": True,
                "error": None
            }
            
        except Exception as e:
            print(f"   ✗ Error converting to DataFrame: {str(e)}")
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
        # If no target statement found, we're done (no retry needed)
        if not state.get("has_target_statement", False):
            return "end"
        
        # If valid table created, we're done
        if state.get("is_valid_table") and state.get("dataframe") is not None:
            return "end"
        
        # If we've reached max attempts, return empty dataframe
        if state["attempt"] >= self.max_attempts:
            print(f"\n   Max attempts ({self.max_attempts}) reached")
            return "max_attempts"
        
        # Otherwise, retry from the beginning
        print(f"\n   Retrying extraction... (attempt {state['attempt']} of {self.max_attempts})")
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
        
        Workflow:
        1. extract_statement_type_and_headers_info: Identify statement and extract headers
        2. normalize_column_headers: Convert headers to standard format
        3. extract_table_with_gpt4o: Extract complete table with normalized headers
        4. convert_to_dataframe: Create pandas DataFrame
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(TableExtractionState)
        
        # Add nodes
        workflow.add_node("extract_statement_type", self._extract_statement_type_and_headers_info)
        workflow.add_node("normalize_headers", self._normalize_column_headers)
        workflow.add_node("extract_table", self._extract_table_with_gpt4o)
        workflow.add_node("convert", self._convert_to_dataframe)
        workflow.add_node("empty_result", self._return_empty_dataframe)
        
        # Set entry point
        workflow.set_entry_point("extract_statement_type")
        
        # Define routing function after extract_statement_type
        def route_after_statement_type(state: TableExtractionState) -> str:
            if state.get("has_target_statement", False):
                return "normalize_headers"
            else:
                return "end"
        
        # Add conditional edge after extract_statement_type
        workflow.add_conditional_edges(
            "extract_statement_type",
            route_after_statement_type,
            {
                "normalize_headers": "normalize_headers",
                "end": "convert"  # Will return empty DataFrame
            }
        )
        
        # Add edges for the main flow
        workflow.add_edge("normalize_headers", "extract_table")
        workflow.add_edge("extract_table", "convert")
        
        # Add conditional edges from convert (for retry logic)
        workflow.add_conditional_edges(
            "convert",
            self._should_retry,
            {
                "end": END,
                "retry": "extract_statement_type",  # Retry from the beginning
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
            "statement_type": StatementType.UNKNOWN.value,
            "has_target_statement": False,
            "original_headers": None,
            "normalized_headers": None,
            "table_data": None,
            "dataframe": None,
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
            print(f"\n{'='*60}")
            print(f"Processing Page {page_num}")
            print(f"{'='*60}")
            
            df, statement_type, confidence = self.extract_from_page(page_image)
            
            if not df.empty and statement_type != StatementType.UNKNOWN:
                financial_stmt = FinancialStatement(df, statement_type, page_num, confidence)
                results[statement_type].append(financial_stmt)
                print(f"\n✓ SUCCESS: Found {statement_type.value} on page {page_num}")
                print(f"  - Confidence: {confidence}")
                print(f"  - Shape: {df.shape}")
            else:
                print(f"\n✗ No target financial statement found on page {page_num}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Balance Sheets found: {len(results[StatementType.BALANCE_SHEET])}")
        for stmt in results[StatementType.BALANCE_SHEET]:
            print(f"  - Page {stmt.page_number}: {stmt.dataframe.shape}")
        print(f"\nIncome Statements found: {len(results[StatementType.INCOME_STATEMENT])}")
        for stmt in results[StatementType.INCOME_STATEMENT]:
            print(f"  - Page {stmt.page_number}: {stmt.dataframe.shape}")
        print(f"\nCash Flow Statements found: {len(results[StatementType.CASH_FLOW])}")
        for stmt in results[StatementType.CASH_FLOW]:
            print(f"  - Page {stmt.page_number}: {stmt.dataframe.shape}")
        print(f"{'='*60}\n")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    extractor = PDFFinancialStatementExtractor(
        api_key="your-api-key-here",
        max_attempts=3,
        dpi=200
    )
    
    # Example: Extract from specific page range
    # results = extractor.extract_from_pdf("apple_10k.pdf", start_page=30, end_page=50)
    
    # Access extracted statements
    # for stmt in results[StatementType.BALANCE_SHEET]:
    #     print(f"\nBalance Sheet from Page {stmt.page_number}:")
    #     print(stmt.dataframe)
    
    print("PDF Financial Statement Extractor ready!")
    print("\nWorkflow: Statement Detection → Header Normalization → Table Extraction → DataFrame Creation")
