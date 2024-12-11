# ### Prerequisite: GhostScript 64-bit version
# #### Python packages: camelot-py, ghostscript, PyPDF2
# 
# **Flow of the script:**
# 1. Parse the PDF for table of contents (standarized across 10-Ks as per sec)
# 2. Read the ToC and then look for a particular item such as "Item 8" and read the page number where it is at
# 3. Go to the page read in step 2 and convert it to a table
# 4. On that table look for the statement such as "Statement of Cashflows" and read it in a table
# 5. Clean the generated table
# 6. Save it as CSV
# 7. Read the CSV and standardize the attributes
# 8. Save the standardized CSV

import argparse

import ghostscript
import camelot
import PyPDF2
import fitz  # PyMuPDF
import re
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# For fuzzy search
from rapidfuzz import process, fuzz

import locale
import os
from datetime import datetime

# In[]
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))
from IPython import get_ipython
if 'ipykernel' in str(get_ipython()):
    script_directory = r"./"

# Set it as the current working directory
os.chdir(script_directory)
print("Current Working Directory:", os.getcwd())

# Download stopwords (if not already done)
nltk.download('stopwords')

class DocumentType(Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    HF = "HF"
    
    @staticmethod
    def from_string(doc_type_str):
        try:
            return DocumentType(doc_type_str)
        except ValueError:
            print(f"Invalid document type: {doc_type_str}")
            return None

class StatementType(Enum):
    Consolidated_Statement_of_CashFlows = 1


# In[]
class PdfUtils:
    CUSTOM_STYLING = [
        {"selector": "th","props": [("background", "#00abe7"),("color", "white"),("font-family", "tahoma"),("text-align", "center"),("font-size", "12px"),],},
        {"selector": "td", "props": [("font-family", "tahoma"), ("color", "black"), ("text-align", "left"), ("font-size", "12px"),],},
        {"selector": "tr:nth-of-type(odd)", "props": [("background", "white"),],},
        {"selector": "tr:nth-of-type(even)", "props": [("background", "#e8e6e6")]},
        {"selector": "tr:hover", "props": [("background-color", "#bfeaf9")]},
        {"selector": "td:hover", "props": [("background-color", "#7fd5f3")]},
    ]


    @staticmethod
    def is_jupyter_notebook():
        try:
            # Jupyter-specific modules
            from IPython import get_ipython
            if 'ipykernel' in str(get_ipython()):
                return True
        except ImportError:
            return False
        return False 
    
    @staticmethod
    def print_df(df):
        if PdfUtils.is_jupyter_notebook():
            s = df.style
            s.set_table_styles(PdfUtils. CUSTOM_STYLING)
            s.hide(axis="index")
            s.format(precision=2)
            s.highlight_null(color='lightgrey')
            # NOTE: Adding div Style with overflow!
            display(HTML("<div style='width: 1000px; overflow: auto;'>" + s.to_html() + "</div>"))
        else:
            print(df)    
        
    @staticmethod
    def print_tables(tables):
        # Iterate over the extracted tables
        for table_index in range(tables.n):
            print(f"\nTable {table_index + 1}:\n")
            # Convert the table to a DataFrame
            PdfUtils.print_df(tables[table_index].df)
            
    @staticmethod
    def pdf_to_text_with_ghostscript(pdf_path, output_text_path, first_page, last_page):
        # Ghostscript arguments to convert a specific page of the PDF to text
        args = [
            "pdf2txt",  # Program name (arbitrary, not used)
            "-dNOPAUSE",  # No pause after each page
            "-dBATCH",  # Exit after processing
            "-sDEVICE=txtwrite",  # Text extraction device
            f"-sOutputFile={output_text_path}",  # Output file
            f"-dFirstPage={first_page}",  # Specify the first page
            f"-dLastPage={last_page}",   # Specify the last page (same as first for single page)
            pdf_path,  # Input PDF file
        ]

        # Run Ghostscript command
        try:
            encoding = locale.getpreferredencoding()
            ghostscript.Ghostscript(*map(str.encode, args))
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
            return None

        # Read the extracted text
        try:
            with open(output_text_path, 'r', encoding='utf-8') as file:
                text_data = file.read()
        except FileNotFoundError:
            print(f"Output text file not found: {output_text_path}")
            return None

        return text_data
    
    @staticmethod
    def get_pdf_page_text_data_using_gs(pdf_path, output_text_path, page):
        output_text_path = "output.txt"  # Temporary text file for output

        # Extract text using Ghostscript
        extracted_text = PdfUtils.pdf_to_text_with_ghostscript(
            pdf_path, output_text_path, page, page)

        # Print the extracted text
        if extracted_text:
            lines = extracted_text.strip().split('\n')
            for line in lines:
                if (len(line) > 100):
                    print(line[:100] + '...')
                else:
                    print(line)

        # Optional: Clean up the output file if needed
        if os.path.exists(output_text_path):
            os.remove(output_text_path)

        return lines
    
    @staticmethod
    def get_deep_copy(df):
        # Create a deep copy of the DataFrame
        deep_copy = df.copy(deep=True)
        return deep_copy

    @staticmethod
    def get_leading_spaces(string):
        leading_spaces = 0
        for char in string:
            if char == ' ':
                leading_spaces += 1
            else:
                break

        return leading_spaces
    
    @staticmethod
    def save_df_as_csv(df, csv_filepathname):
        df.to_csv(csv_filepathname, encoding='utf-8', index=False, header=True)   
        
    @staticmethod
    def count_columns_onwards(df, col_name):
        # Find the index of 'Page' column
        page_index = df.columns.get_loc(col_name)
        
        # Calculate the number of columns after col_name including the col_name
        columns_after_page = len(df.columns) - page_index
        return columns_after_page
        


class PdfTableReader:
    # Constant declaration
    MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE = 15
    MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT = 25
    MAX_ROWS_TO_SCAN_FOR_COLUMN_META_INFO = 5
    def __init__(self, inpath, outpath, filename):
        if not filename:
            raise ValueError("Filename cannot be empty. Please check and pass a valid PDF file")
    
        if not inpath:
            inpath = "./"
            
        if not outpath:
            outpath = "./"
            
        self.inpath = inpath
        self.outpath = outpath
        
        self.page_offset = 0
        self.total_pages = 0
        
        filename, ext = os.path.splitext(filename)
        if ext and ext.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {ext}. Only PDF files are allowed.")
        
        ext = ".pdf"
        self.filename = filename 
        
        self.pdf_filepath = os.path.join(self.inpath, f"{filename}{ext}")

        if not os.path.exists(self.pdf_filepath):
            raise ValueError(f"File does not exist at location: {self.pdf_filepath}")
 
        if not os.path.isdir(self.outpath):
            raise ValueError(f"Output directory does not exist at location: {self.inpath}")
        
        self.temp_path = os.path.join(self.outpath, "temp")
        if not os.path.isdir(self.temp_path):
            os.makedirs(self.temp_path)
            print(f"Directory '{self.temp_path}' created.")
            
        self.csv_filepath = os.path.join(self.temp_path, f"{filename}.csv")
        self.std_filepath = os.path.join(self.outpath, f"std_{filename}.csv")
            
        print(f"input: {self.pdf_filepath}, output will be generated at: {self.outpath}")
        
    def get_input_filename(self):
        return self.pdf_filepath
    
    def get_temp_path(self):
        return self.temp_path
    
    def get_temp_csv_filepathname(self):
        return self.csv_filepath
    
    def get_standard_csv_filepathname(self):
        return self.std_filepath
    
    def get_total_pages(self):
        if self.total_pages == 0:
            pdf_document = fitz.open(self.pdf_filepath)
            # Return the total number of pages
            self.total_pages = pdf_document.page_count
            
        return self.total_pages
    
    def calculate_page_offset(self):
        try:
            # Open the PDF file in binary mode
            with open(self.pdf_filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Get the total number of pages
                self.total_pages = len(reader.pages)
                
                if self.total_pages == 0:
                    return "PDF has no pages"
                
                # Find the middle page
                middle_page = self.total_pages // 2
                
                # Extract text from the middle page (note: pages are zero-indexed)
                page = reader.pages[middle_page]
                text = page.extract_text()
                if not text:
                    return "No text could be extracted from the middle page."
    
                # Split the extracted text into lines
                lines = text.splitlines()
                if not lines:
                    return "Extracted text from the middle page is empty."
                
                # Initialize the page number text as None
                page_number_text = None
                
                # Assuming the page number is at the bottom, we get the last non-empty line
                for line in reversed(lines):
                    if line.strip():  # Skip empty lines
                        page_number_text = line.strip()
                        break
                
                # Check if we successfully extracted the page number
                if page_number_text is None:
                    return "Page number could not be determined from text"
                
                # Check if the extracted text is a valid number
                if page_number_text.isdigit():
                    extracted_page_number = int(page_number_text)
                    expected_page_number = middle_page + 1
                    
                    # Compare extracted page number with the expected page number
                    if extracted_page_number != expected_page_number:
                        self.page_offset = expected_page_number - extracted_page_number
                        return f"Page number is different. Expected: {expected_page_number}, Extracted: {extracted_page_number}. Difference: {self.page_offset}"
                    else:
                        return f"Page number matches the expected value: {expected_page_number}"
                else:
                    return "Page number could not be determined or is not numeric"
                
        except FileNotFoundError:
            return f"File '{self.pdf_filepath}' was not found."
        except PyPDF2.errors.PdfReadError:
            return "Failed to read PDF. The file may be corrupted or not a valid PDF."
        except Exception as e:
            return f"An unexpected error occurred: {e}"
    
    # Function to search Table of Contents for a string in the DataFrame
    def search_toc(self, df, search_string):
        # Check if the search_string is in any of the DataFrame's string columns
        result = df[df.apply(lambda row: row.astype(str).str.contains(search_string).any(), axis=1)]
        
        if not result.empty:
            # Extract the value from the last column (Numerical column in this case)
            page_no = result.iloc[0, -1]
            print(f"Found row: \n{result}")
            print(f"Numerical value from the last column: {page_no}")
            return page_no
        else:
            print("search_toc(): No matching rows found.")
            return None

    # Search item specific table that is obtained from search_toc
    def search_itemized_table(self, df, string1, string2, substatement):
        # Search for rows where both string1 and string2 are found in the same row
        result = df[df.apply(lambda row: (row.astype(str).str.contains(string1).any() and
                                          row.astype(str).str.contains(string2).any()), axis=1)]
        
        if not result.empty:
            if substatement is not None:
                # Step 2: Get the index of the first matching row
                start_index = result.index[0]  #row where parent item is found
                
                # Step 3: Filter rows starting from start_index + 1
                subsequent_df = df.iloc[start_index + 1:]
                
                # Step 4: Search for substatement in the subsequent rows
                sub_result = subsequent_df[subsequent_df.apply(lambda row: any(substatement in str(cell) for cell in row), axis=1)]
                if not sub_result.empty:
                    print(f"{substatement} found in subsequent rows of main table of contents")
                    print(sub_result)
                    page_no = sub_result.iloc[0, -1]
                    print(f"Numerical value from the last column: {page_no}")
                    return page_no, True
            
            # Extract the value from the last column (Numerical column in this case)
            page_no = result.iloc[0, -1]
            print(f"Found row: \n{result}")
            print(f"Numerical value from the last column: {page_no}")
            return page_no, False
        else:
            print(f"search_itemized_table(): No matching rows found for {string1} and {string2}")
            return None, None
  
    def ensure_integer(self, value):
        if isinstance(value, int):  # Check if the variable is already an integer
            return value
        try:
            return int(value)  # Attempt to convert to an integer
        except ValueError:
            raise ValueError(f"Cannot convert {value} to an integer.")  # Handle invalid conversion
  
    def ensure_integer_or_assign_zero(self, value):
        if isinstance(value, int):  # Check if the variable is already an integer
            return value
        try:
            return int(value)  # Attempt to convert to an integer
        except ValueError:
            return 0
        
        return 0


    def get_hyperlinked_page(self, page, search_text):
        doc = fitz.open(self.pdf_filepath)

        # Extract links from the first page
        index_page = doc[page - 1] #fitz is zero based

        rects = index_page.search_for(search_text)
        
        if rects:
            # Extract all links from the page
            links = index_page.get_links()
            if links:
                for rect in rects:
                    for link in links:
                        link_rect = fitz.Rect(link["from"])
                        if rect.intersects(link_rect):
                            print(f"Found link associated with '{search_text}': {link}")
                            if "page" in link:
                                print(f"Link points to page: {link['page']}")
                                return self.ensure_integer(link['page'])

        return 0
  
    def get_adjusted_page_REMOVE(self, toc_page, search_text):
        doc = fitz.open(self.pdf_filepath)

        # Extract links from the first page
        index_page = doc[toc_page - 1] #fitz is zero based

        rects = index_page.search_for(search_text)
        
        if rects:
            # Extract all links from the page
            links = index_page.get_links()
            if links:
                for rect in rects:
                    for link in links:
                        link_rect = fitz.Rect(link["from"])
                        if rect.intersects(link_rect):
                            print(f"Found link associated with '{search_text}': {link}")
                            if "page" in link:
                                print(f"Link points to page: {link['page']}")
                                return self.ensure_integer(link['page'])

        return 0
  
    def scan_adjacent_pages(self, page, text):
        scanned_page = self.scan_page_for_content(page, text, 1)       
        if(scanned_page > 0):
            return scanned_page
        else:
            scanned_page = self.scan_page_for_content(page - 1, text, 1)       
            if(scanned_page > 0):
                return scanned_page
            else:
                scanned_page = self.scan_page_for_content(page + 1, text, 1)       
                if(scanned_page > 0):
                    return scanned_page
            
        return 0
    
    def scan_for_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(self, page_no, ignore_spaces=False):
        text = "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"
        scanned_page = self.scan_page_for_content(page_no, text, 1, ignore_spaces)       
        if(scanned_page == 0):
            scanned_page = self.scan_page_for_content(page_no - 1, text, 1, ignore_spaces)       
            if(scanned_page == 0):
                scanned_page = self.scan_page_for_content(page_no + 1, text, self.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE, ignore_spaces)       
                if(scanned_page == 0):
                    print(f"{text} not found on pages {page_no-1}, {page_no}, {page_no+2}")
                    print("Assuming there is no sub table")
        
        if scanned_page == 0 and ignore_spaces == False:
            scanned_page = self.scan_for_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(page_no, True)
            return scanned_page
        
        return scanned_page

    
    def get_page_from_the_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(self, page_number, statement):
        tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_number), flavor='stream')

        statement_page = 0
        # Iterate over the extracted tables
        for table_index in range(tables.n):
            # Convert the table to a DataFrame
            df = tables[table_index].df
            if((df.shape[0] == 0) or (df.shape[1] < 2)):
                continue

            print(f"PRINTING FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table with index {table_index}")
            PdfUtils.print_df(df)
            
            # Define the patterns to search for
            patterns = ["Consolidated Statements of", "Consolidated Balance Sheets"]
            
            # Use apply() to search across entire rows
            matching_rows = df.apply(
                lambda row: row.astype(str).str.contains("|".join(patterns), case=False, regex=True).any(),
                axis=1
            )
            
            # Count the rows that match
            count = matching_rows.sum()
            
            if count < 2:
                continue
            
            content_page = self.search_toc(df, statement)
            
            print(f"content_page is {content_page}")
            if( content_page is not None):
                statement_page = self.get_hyperlinked_page(page_number, statement)
                if statement_page == 0:
                    try:
                        statement_page = int(content_page)  # Attempt to convert the string to an integer
                        #page = page + self.page_offset
                        #scanned_page = self.scan_page_for_content(page, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                        #if(scanned_page > 0):
                        ##    page = scanned_page
                    except ValueError:
                        statement_page = 0
                        # Look for page in subsequent pages
                        #page = self.scan_page_for_content(page_number + 1, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                #else:
                #    scanned_page = self.scan_page_for_content(page, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                #    if(scanned_page > 0):
                #        page = scanned_page
                
            return True, statement_page

        return False, statement_page
    
    # Function to extract text from the PDF one page at a time and look for the required parts
    def find_page_for_item_x_with_parts(self, item, statement_type, substatement):
        page_number = 1
        adjusted_page = 0
        # Iterate through each page individually
        while True:
            try:
                if page_number > PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE:
                    print("No table found in first {}".format(PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE))
                    return 0
                
                try:
                    # Extract tables from the PDF
                    tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_number), flavor='stream', edge_tol=0)
                except Exception as e:
                    print(f"There was error processing page {page_number}: {e}")
                    # Increment page number to continue search
                    page_number += 1
                    continue

                    
                print(f"Total tables extracted from page {page_number}: {tables.n}")

                # Iterate over the extracted tables
                for table_index in range(tables.n):
                    # Convert the table to a DataFrame
                    df = tables[table_index].df

                    if(df.shape[0] == 0 or df.shape[1] == 0):
                        continue
                    
                    # Step 1:
                    result = df[df.apply(lambda row: (row.astype(str).str.contains(item).any() and
                                                      row.astype(str).str.contains(statement_type).any()), axis=1)]
                    
                    result = df[df.apply(lambda row: (
                            row.astype(str).str.contains(item, case=False).any() and
                            row.astype(str).str.contains(statement_type, case=False).any()
                            ), axis=1)]
                                            
                    if result.empty:
                        continue
                    
                    # Found "Item 8: Financial Statements and Supplementary Data"; this is our main ToC page
                    # We are here means that DF has Item (ex 8) and also the statement_type (ex: Financial Statements and Supplementary Data)
                    # Lets continue looking into it further

                    if substatement is None:
                        print(f"WARNING: substatement passed is None, such cases are not yet handled, bailing ouy!!!")
                        break
                        
                    sub_table_page_no = 0
                    sub_stmt_page_no = 0
                    
                    #Process for "Consolidated Statements of Cash Flows"
                    # Step 2: Get the index of the first matching row
                    start_index = result.index[0]  #row where parent item is found
                    
                    # Step 3: Filter rows starting from start_index + 1
                    subsequent_df = df.iloc[start_index + 1:]
                    
                    nested = False
                    # Step 4: Search for substatement in the subsequent rows
                    sub_result = subsequent_df[subsequent_df.apply(lambda row: any(substatement in str(cell) for cell in row), axis=1)]
                    if not sub_result.empty:
                        print(f"{substatement} found in subsequent rows of main table of contents")
                        sub_stmt_page_no = self.ensure_integer_or_assign_zero(sub_result.iloc[0, -1])
                        print(f"value of substatement {substatement} from the last column: {sub_stmt_page_no}")
                        nested = True
                    else:
                        # Extract the value from the last column (Numerical column in this case)
                        sub_table_page_no = self.ensure_integer_or_assign_zero(result.iloc[0, -1])
                        print(f"value of the subtable page from the last column: {sub_table_page_no}")
                    

                    """
                    Several scenarios here
                    1) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X has a subtable containing the entry "Consolidated Statements of Cash Flows"
                        
                    2) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X doesnot have a subtable containing the entry "Consolidated Statements of Cash Flows"
                        instead it points to page Y that has subtable
 
                    3) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X doesnot have a subtable containing the entry "Consolidated Statements of Cash Flows"
                        instead subsequent pages have the relevalt full tables, one of them for "Consolidated Statements of Cash Flows"
                    
                    4) Toc Page has Item 8: Financial Statements and Supplementary Data 
                        and underneatn Toc itself is subtable with entry "Consolidated Statements of Cash Flows"
                        
                        
                    Flow
                    does Toc has Item 8: Financial Statements and Supplementary Data?
                    Yes:
                        does Toc has subtable too?
                        Yes: go read the "Consolidated Statements of Cash Flows" entry and figure out the page Y
                        No:
                            go to page X and look for "Financial Statements and Supplementary Data"
                            Found?
                            Yes: Look for Subtable
                                Found Subtable?
                                Yes: Good, read the stmt of cashflows in that table and figure out page Y
                                No: Keep reading subsequent pages till you find stmt of cashflows and figure out page Y
                    """

                    # Secnario 1: Nested subtable is found in main ToC
                    if nested is True:
                        hyperlinked_sub_stmt_page_no = self.get_hyperlinked_page(page_number, substatement)
                        if hyperlinked_sub_stmt_page_no == 0:
                            hyperlinked_sub_stmt_page_no = sub_stmt_page_no #Use the one from ToC
                        
                        if hyperlinked_sub_stmt_page_no == 0: #still zero means we failed
                            print(f"WARNING: The page couldnt be determined from ToC or using links, bailing out!!!")
                        
                        return hyperlinked_sub_stmt_page_no
                        
                    # There are 3 further secanrios here
                    hyperlinked_sub_table_page_no = self.get_hyperlinked_page(page_number, statement_type)
                    if hyperlinked_sub_table_page_no > 0:
                        sub_table_page_no = hyperlinked_sub_table_page_no
                        
                    #The page of interest is hyperlinked_sub_table_page_no
                    #it may contain a subtable or a link to subtable or subtable in the vicinity
                    #or just direct statements that are supposed to be in subtable
                    
                    #Lets scan for subtable
                    scanned_page = self.scan_for_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(sub_table_page_no)
                    if scanned_page == 0:
                        
                        print(f"WARNING: The page couldnt be determined from ToC, bailing out!!!")
                        return sub_table_page_no

                    sub_table_page_no = scanned_page
                    # 3 scenarios here
                    # Look for subtable next

                    # Scenario 2: Sub-Table exists
                    table_found, statement_page = self.get_page_from_the_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(
                        sub_table_page_no, substatement)
                    
                    if table_found is True:
                        return statement_page
                    
                    # Scenario 3: There is no subtable
                    # Scan subsequent pages for the stament directly
                    scanned_page = self.scan_page_for_content(scanned_page, substatement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE)
                    if scanned_page > 0:
                        #May be it is part of sub-table, lets check it again
                        table_found, statement_page = self.get_page_from_the_FINANCIAL_STATEMENTS_AND_SUPPLEMENTARY_DATA_Table(
                            scanned_page, substatement)
                        if table_found is True:
                            return statement_page
                        
                    # We are here means that sub-table is not found and instead we just ran into the statement by search
                    return scanned_page
                
                # Increment page number to continue search
                page_number += 1
            
            except Exception as e:
                print(f"Reached the end of the PDF or encountered an error: {e}")
                break
        
        print("The relevant page was not found.")
        return 0  
    
    # Function to extract text from the PDF one page at a time and look for the required parts
    def find_page_for_item_x_with_parts_bak(self, item, statement_type, substatement):
        page_number = 1
        adjusted_page = 0
        # Iterate through each page individually
        while True:
            try:
                if page_number > PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE:
                    print("No table found in first {}".format(PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE))
                    return None, 0, False

                # Extract tables from the PDF
                tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_number), flavor='stream', edge_tol=0)
            
                print(f"Total tables extracted from page {page_number}: {tables.n}")

                # Iterate over the extracted tables
                for table_index in range(tables.n):
                    # Convert the table to a DataFrame
                    df = tables[table_index].df


                    """
                    Several scenarios here
                    1) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X has a subtable containing the entry "Consolidated Statements of Cash Flows"
                        
                    2) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X doesnot have a subtable containing the entry "Consolidated Statements of Cash Flows"
                        instead it points to page Y that has subtable
 
                    3) Toc Page has Item 8: Financial Statements and Supplementary Data - X
                        X doesnot have a subtable containing the entry "Consolidated Statements of Cash Flows"
                        instead subsequent pages have the relevalt full tables, one of them for "Consolidated Statements of Cash Flows"
                    
                    4) Toc Page has Item 8: Financial Statements and Supplementary Data 
                        and underneatn Toc itself is subtable with entry "Consolidated Statements of Cash Flows"
                        
                        
                    Flow
                    does Toc has Item 8: Financial Statements and Supplementary Data?
                    Yes:
                        does Toc has subtable too?
                        Yes: go read the "Consolidated Statements of Cash Flows" entry and figure out the page Y
                        No:
                            go to page X and look for "Financial Statements and Supplementary Data"
                            Found?
                            Yes: Look for Subtable
                                Found Subtable?
                                Yes: Good, read the stmt of cashflows in that table and figure out page Y
                                No: Keep reading subsequent pages till you find stmt of cashflows and figure out page Y
                    """

                    content_page, nested = self.search_itemized_table(df, item, statement_type, substatement)
                    
                    if content_page is not None:
                        search_text = statement_type
                        if nested == True:
                            search_text = substatement
                        try:
                            adjusted_page = self.get_adjusted_page(page_number, search_text)
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                            adjusted_page = 0
                            
                        if adjusted_page == 0:
                            adjusted_page = content_page
                        
                        scan_page = adjusted_page
                        
                        scanned_page = self.scan_adjacent_pages(scan_page, substatement)
                        if( scanned_page == 0):
                            scanned_page = self.scan_page_for_content(scan_page + 2, substatement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE)

                        if( scanned_page > 0):
                            self.page_offset = scanned_page - int(content_page)
                            return content_page, scanned_page, nested
                        
                        return None, 0, False
                # Increment page number to continue search
                page_number += 1
            
            except Exception as e:
                print(f"Reached the end of the PDF or encountered an error: {e}")
                break
        
        print("The relevant page with Part I, Part II, Part III, and Item 8 was not found.")
        return None, 0, False  
  
    # Function to extract text from the PDF one page at a time and look for the required parts
    def find_item_x_with_parts(self, item, statement_type):
        page_number = 1

        # Iterate through each page individually
        while True:
            try:
                if page_number > PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE:
                    print("No table found in first {}".format(PdfTableReader.MAX_PAGES_TO_SCAN_FOR_INDEX_TABLE))
                    return None

                # Extract tables from the PDF
                tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_number), flavor='stream')
            
                print(f"Total tables extracted from page {page_number}: {tables.n}")

                # Iterate over the extracted tables
                for table_index in range(tables.n):
                    # Convert the table to a DataFrame
                    df = tables[table_index].df

                    if(item is None):
                        content_page = self.search_toc(df, statement_type)
                    else:
                        content_page = self.search_itemized_table(df, item, statement_type)
                    
                    if( content_page is not None and int(content_page) > page_number ):
                        content_page_tables = camelot.read_pdf(self.pdf_filepath, pages=str(int(content_page) + self.page_offset), flavor='stream')
                        
                        print(f"\nTable {table_index + 1}:\n")
        
                        PdfUtils.print_df(df)
                        
                        return content_page_tables
                         
                # Increment page number to continue search
                page_number += 1
            
            except Exception as e:
                print(f"Reached the end of the PDF or encountered an error: {e}")
                break
        
        print("The relevant page with Part I, Part II, Part III, and Item 8 was not found.")
        return None
    
    # Compare table structure
    def are_tables_continuous(self, table1, table2):
        table1_minus_page = table1.drop('Page', axis=1)
        table2_minus_page = table2.drop('Page', axis=1)
        
        print(f"table1.shape[1] is {table1_minus_page.shape[1]}")
        print(f"table2.shape[1] is {table2_minus_page.shape[1]}")

        if table1_minus_page.shape[1] != table2_minus_page.shape[1]:
            return False

        print("Printing table1.iloc[0]")
        print(table1_minus_page.iloc[0])

        print("Printing table2.iloc[0]")
        print(table2_minus_page.iloc[0])

        if (table1_minus_page.iloc[0] == table2_minus_page.iloc[0]).all():
            return True

        return False


    def search_with_statement_pattern(self, text, search_string):
        # Build the dynamic regex pattern
        # Replace "Statement" with "Statement[s]?" to account for "Statement" and "Statements"
        pattern = re.sub(r"\bStatements?\b", r"Statement[s]?", search_string, flags=re.IGNORECASE)
        # Perform a case-insensitive search
        return re.search(pattern, text, re.IGNORECASE) is not None

    def search_text_with_spaces_ignored(self, target, text):
        # Remove all spaces and convert both strings to lowercase
        normalized_target = re.sub(r"\s+", "", target).lower()
        normalized_text = re.sub(r"\s+", "", text).lower()
        # Check if the normalized target exists in the normalized text
        return normalized_target in normalized_text


    def check_page_for_data(self, page, search_text, ignore_spaces=False):
        output_text_path = os.path.join(self.temp_path, f"{self.filename}-gs.txt")

        page_text = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, page, page)

        if ignore_spaces:
            return self.search_text_with_spaces_ignored(search_text, page_text)
        else:
            return self.search_with_statement_pattern(page_text, search_text)

    def check_page_for_data_list(self, page, list_text):
        output_text_path = os.path.join(self.temp_path, f"{self.filename}-gs.txt")

        page_text = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, page, page)

        # Check if every item in list_text is in page_text, ignoring case
        return all(text.lower() in page_text.lower() for text in list_text)


    def are_consecutive_pages(self, first_page, next_page, statement_type):
        output_text_path = os.path.join(self.temp_path, f"{self.filename}-gs.txt")

        page_text_1 = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, first_page, first_page)
        page_text_2 = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, next_page, next_page)

        if(statement_type.lower() in page_text_1.lower()):
            if(statement_type.lower() in page_text_2.lower()):
                print(f"are_consecutive_pages retruns true as both pages has >>{statement_type}<< in it")
                return True

        print(f"are_consecutive_pages returns false as both pages dont have >>{statement_type}<< in it")
        return False

    # Use Camelot to extract tables
    def extract_table_from_page(self, page_num):
        tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_num), flavor='stream', edge_tol=100)
        PdfUtils.print_tables(tables)
        if len(tables) > 0:
            df = tables[0].df
            df['Page'] = page_num            
        else:
            df = None
            
        return df

    # Full pipeline to check if two tables are continuous
    def check_and_merge_tables(self, page, statement_type):
        dfs = []
        
        min_cols = 3
        min_rows = 5       
        # Extract tables from both pages
        if "Consolidated Statements of Cash Flows" == statement_type:
            min_rows = 15
            
        check_if_continuous = True
        table_page = self.extract_table_from_page(page)
        if table_page.shape[0] > min_rows and table_page.shape[1] > min_cols:
            dfs.append(table_page)

        current_page = page + 1

        while check_if_continuous == True and current_page <= page + 5:
            table_page = self.extract_table_from_page(current_page)
            if len(dfs) > 0:
                check_if_continuous = self.are_consecutive_pages(current_page, current_page - 1, statement_type)
                if check_if_continuous is False:
                    # Check if tables are structurally continuous
                    check_if_continuous = self.are_tables_continuous(dfs[-1], table_page)
                
                if check_if_continuous is True:
                    print(f"are_tables_continuous returns {check_if_continuous} for pages {current_page - 1} and {current_page}")
                    dfs.append(table_page)
            else:
                dfs.append(table_page)
                
            current_page = current_page + 1
        
        """
        table_page1 = self.extract_table_from_page(page)
        table_page2 = self.extract_table_from_page(page + 1)

        table_continuous = self.are_consecutive_pages(page, page + 1, statement_type)
            
        if table_continuous is False:
            # Check if tables are structurally continuous
            table_continuous = self.are_tables_continuous(table_page1, table_page2)
            print(f"are_tables_continuous returns {table_continuous} for pages {page} and {page+1}")

        dfs.append(table_page1)

        # If both text and table structure indicate continuity, merge tables
        if table_continuous is True:
            # Merge tables (exclude headers from the second table)
            dfs.append(table_page2)
        """
        return dfs

    def get_table_as_df(self, page, statement_type):
        return self.check_and_merge_tables(page, statement_type)

    def find_statement(self, tables, statement_type):
        #print_tables(tables)
        # Iterate over the extracted tables
        for table_index in range(tables.n):
            # Convert the table to a DataFrame
            df = tables[table_index].df
            print(f"PRINTING TABLE with index {table_index}")
            PdfUtils.print_df(df)
            content_page = self.search_toc(df, statement_type)

            print(f"content_page is {content_page}")
            if( content_page is not None):
                return self.get_table_as_df(int(content_page) + self.page_offset, statement_type)

    def scan_page_for_content(self, page_number, search_text, number_of_pages_to_scan, ignore_spaces=False):
        scan_page = page_number
        max_pages_to_scan = page_number + number_of_pages_to_scan
        if(self.get_total_pages() < max_pages_to_scan + 1):
            max_pages_to_scan = self.get_total_pages() - 1
            
        while scan_page < max_pages_to_scan:
            isFound = self.check_page_for_data(scan_page, search_text, ignore_spaces)
            if isFound is True:
                return scan_page
            
            scan_page = scan_page + 1  
        
        return 0
    
    def scan_page_for_statement_table(self, page_number, list_text, number_of_pages_to_scan):
        scan_page = page_number
        max_pages_to_scan = page_number + number_of_pages_to_scan
        if(self.get_total_pages() < max_pages_to_scan + 1):
            max_pages_to_scan = self.get_total_pages() - 1
            
        output_text_path = os.path.join(self.temp_path, f"{self.filename}-gs.txt")

        while scan_page < max_pages_to_scan:
            page_text = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, scan_page, scan_page)

            if self.search_with_statement_pattern(page_text, list_text[0]) is True:                    
                # Check if every item in list_text is in page_text, ignoring case
                bFound = all(text.lower() in page_text.lower() for text in list_text[1:])
                if bFound is True:
                    return scan_page     
                else:
                    # see if subsequent page has data
                    scan_page += 1
                    page_text = PdfUtils.pdf_to_text_with_ghostscript(self.pdf_filepath, output_text_path, scan_page, scan_page)
                    bFound = all(text.lower() in page_text.lower() for text in list_text[1:])
                    if bFound is True:
                        return scan_page
                    
                    continue
            
            scan_page = scan_page + 1  
        
        return scan_page

    def find_statement_from_subtable(self, page_number, statement):  
        tables = camelot.read_pdf(self.pdf_filepath, pages=str(page_number), flavor='stream')

        content_page = None
        # Iterate over the extracted tables
        for table_index in range(tables.n):
            # Convert the table to a DataFrame
            df = tables[table_index].df
            print(f"PRINTING TABLE with index {table_index}")
            PdfUtils.print_df(df)
            content_page = self.search_toc(df, statement)

            print(f"content_page is {content_page}")
            if( content_page is not None):
                page = self.get_adjusted_page(page_number, statement)
                if page == 0:
                    try:
                        page = int(content_page)  # Attempt to convert the string to an integer
                        page = page + self.page_offset
                        scanned_page = self.scan_page_for_content(page, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                        if(scanned_page > 0):
                            page = scanned_page
                    except ValueError:
                        # Look for page in subsequent pages
                        page = self.scan_page_for_content(page_number + 1, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                else:
                    scanned_page = self.scan_page_for_content(page, statement, PdfTableReader.MAX_PAGES_TO_SCAN_FOR_CONSOLIDATED_STATEMENT)       
                    if(scanned_page > 0):
                        page = scanned_page
        
        if(page > 0):
            return self.get_table_as_df(page, statement)
        
        return None

    
    def get_item8_statement(self, statement):
        # usage
        item = "8"
        statement_type = "Financial Statements and Supplementary Data"

        #page_offset_result = self.get_page_offset()
        #print(page_offset_result)
        
        # Step: Find the page with Part I, Part II, Part III, and Item 8, and return the table as a DataFrame
        #extracted_tables, nested = self.find_item_x_with_parts(item, statement_type, statement)
        
        statement_page = self.find_page_for_item_x_with_parts(item, statement_type, statement)
        if(statement_page > 0):
            list_text = []
            list_text.append(statement)
            list_text.append('Operating')
            list_text.append('Net')

            try:
                scanned_page = self.scan_page_for_statement_table(statement_page, list_text, 100)   
                if scanned_page > 0:
                    dfs = self.get_table_as_df(int(scanned_page), statement)
                        
                    return dfs
            except Exception as e:
                print(e)
        
        return None

    def get_item8_statement_bak(self, statement):
        # usage
        item = "8"
        statement_type = "Financial Statements and Supplementary Data"

        #page_offset_result = self.get_page_offset()
        #print(page_offset_result)
        
        # Step: Find the page with Part I, Part II, Part III, and Item 8, and return the table as a DataFrame
        #extracted_tables, nested = self.find_item_x_with_parts(item, statement_type, statement)
        
        content_page, adjusted_page, nested = self.find_page_for_item_x_with_parts(item, statement_type, statement)
        if(adjusted_page == 0):
            adjusted_page = content_page
        
        print(f'Page offset is {self.page_offset}')

        if content_page is not None:
            dfs = []
            
            if nested is True:
                dfs = self.get_table_as_df(int(adjusted_page), statement if nested is True else statement_type)
            else: 
                dfs = self.find_statement_from_subtable(adjusted_page, statement)
                
            return dfs
        
        return None
  
    def scan_first2_rows_and_move_text_to_first_column(self, df):
        # Replace all cells that contain only spaces with NaN
        # df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        # Replace in columns from the 2nd column onward
        df.iloc[:, 1:] = df.iloc[:, 1:].replace(r'^\s*$', np.nan, regex=True)
        for i in range(min(2, len(df))):  # Checking only the first two rows
            # Select columns excluding the first one (index 0)
            non_first_column_texts = df.iloc[i, 1:].dropna()

            if len(non_first_column_texts) == 1:  # If there's text in exactly one non-first column
                text_column = non_first_column_texts.index[0]  # Get the column index with text
                df.iat[i, 0] = df.iat[i, text_column]  # Move text to the first column (index 0)
                df.iat[i, text_column] = None  # Clear the original column
        
        return df

    # Function to check if a string is entirely non-alphanumeric
    def is_non_alphanumeric_or_empty(self, text):
        # Strip leading/trailing spaces and check if the text is empty or non-alphanumeric
        return all(not char.isalnum() for char in str(text)) or str(text).strip() == '' or pd.isna(text)

    # Function to drop columns if all their values are non-alphanumeric or empty
    def drop_columns_with_all_non_alphanumeric(self, df):
        # Iterate over each column
        for column in df.columns:
            # Check if all cells in the column are either non-alphanumeric or empty
            if df[column].apply(self.is_non_alphanumeric_or_empty).all():
                df.drop(columns=[column], inplace=True)
    
    def clean_data(self, df):
        # Call the function to drop columns with entirely non-alphanumeric cells
        df = self.scan_first2_rows_and_move_text_to_first_column(df)
        self.drop_columns_with_all_non_alphanumeric(df)
        #df.replace(r'^\-+$', np.nan, regex=True, inplace=True)
        #df.replace(r'^\—+$', np.nan, regex=True, inplace=True)
        
        df.iloc[:, 1:].replace(r'^\s*$', np.nan, regex=True, inplace=True)
        #df.iloc[:, 1:] = df.iloc[:, 1:].replace(r'^\s*$', np.nan, regex=True, inplace=True)

     

    def find_first_row_with_202x(self, df, first_col_name):
        # Filtering rows where the first column is NaN and any subsequent column has a year 202x
        mask = df[first_col_name].apply(lambda x: pd.isna(x) or str(x).strip() == "") & df.iloc[:, 1:].apply(lambda x: x.str.contains(r'202\d').any(), axis=1)
        
        # Get the first matching row with the index (row number)
        first_matching_row = df[mask].head(1)
        
        if not first_matching_row.empty:
            row_number = first_matching_row.index[0]
            
            # Check if all valid (non-NaN) cells are in the format 202x in the row
            row_data = first_matching_row.iloc[0, 1:]  # Exclude the first column
            all_202x = row_data.dropna().apply(lambda x: bool(pd.Series(x).str.fullmatch(r'.*\b202\d\b').any()))
            
            if all(all_202x):
                return True, row_number
            
        return False, None


    def concatenate_single_non_nan_rows(self, df, row_number):
        # Initialize an empty list to collect valid values
        concatenated_values = []
        rows_to_be_dropped = []
        print(f"row number passed is {row_number}")
        # Iterate through rows before the given row number
        for i in range(row_number):
            row = df.iloc[i]

            print(f"Processing row number {i}")
            # Drop NaN values and count how many valid values remain
            non_nan_values = row.dropna()
            print(f"Item value in {i} is {non_nan_values.iloc[0]}")

            # If the row has exactly one non-NaN value, add it to the list
            if len(non_nan_values) == 1:
                concatenated_values.append(non_nan_values.iloc[0])
                rows_to_be_dropped.append(i)
                
        # Concatenate the collected values with '|' separator
        if concatenated_values:
            print(concatenated_values)
            while(len(rows_to_be_dropped) > 0):
                rowindex = rows_to_be_dropped.pop()
                df.drop(index=rowindex, inplace=True)

            return '|'.join(map(str, concatenated_values))
        else:
            return None


    def concatenate_value_to_row(self, df, row_number, value, col):
        df.at[row_number, col] = value
        return df


    # ### Fix the column headers, extract year information and other information before the Year row
    # ### Process/Attach the group information to each row
    def fix_column_meta_info(self, df, items_col):
        result, row_number = self.find_first_row_with_202x(df, items_col)
        
        print(f"Result: {result}")
        if result:
            print(f"Matching row number: {row_number}")
            if(row_number <= PdfTableReader.MAX_ROWS_TO_SCAN_FOR_COLUMN_META_INFO): #Found years in first 5 rows
                str_col_info = self.concatenate_single_non_nan_rows(df, row_number)
        
                if str_col_info:
                    print(f"Concatenated values: {str_col_info}")
                    df = self.concatenate_value_to_row(df, row_number, str_col_info, items_col)
                else:
                    print("No rows with a single non-NaN value found.")
                
        else:
            print("No matching row found.")

    def clean_pages(self, df_list):
        cleaned_dfs = []
        current_df = 0
        items_col = 0
        for page in df_list:
            page_value = page.loc[0, 'Page']
            page_df = page.drop('Page', axis=1)

            self.clean_data(page_df)
            current_df = current_df + 1

            if current_df == 1:
                # Fix the column meta information
                self.fix_column_meta_info(page_df, items_col)
                page_df.columns = range(page_df.shape[1])
                page_df.reset_index(drop=True, inplace=True)
                page_df['Page'] = page_value #add page number back

                cleaned_dfs.append(page_df)
            else:
                result, start_index = self.find_first_row_with_202x(page_df, items_col)
                if result is True:
                    # Extract rows from the given row index onwards
                    page_df = page_df.iloc[start_index + 1:]
                    page_df.columns = range(page_df.shape[1])
                    page_df.reset_index(drop=True, inplace=True)
                    page_df['Page'] = page_value #add page number back

                    cleaned_dfs.append(page_df)
            
        return cleaned_dfs

    def merge_dfs(self, df_list):
        df = df_list[0]
        if len(df_list) > 1:
            for page_index in range(1, len(df_list)):
                df_page = df_list[page_index]
                #df_no_header = df_page.iloc[1:]
                df = pd.concat([df, df_page]).reset_index(drop=True)
            
        print(df.shape)
        return df

    
    def get_consolidated_statement_of_cashflows(self):
        dfs = self.get_item8_statement(statement = "Consolidated Statements of Cash Flows")
        
        # The dfs is a list of DFs that hold raw pages for a statement
        # This needs to be claened and merged
        # Clean the DFs
        cleaned_df_pages = self.clean_pages(dfs)
        for df in cleaned_df_pages:
            PdfUtils.print_df(df)

        # Merge together
        statement_df = self.merge_dfs(cleaned_df_pages)
        # Rename the first column to "Items"
        items_col = "Items"
        statement_df.rename(columns={statement_df.columns[0]: items_col}, inplace=True)
        PdfUtils.print_df(statement_df)
            
        return statement_df


class StatementGroupProcessor:
    # ### Process/Attach the group information to each row
    def __init__(self, pdf_filepath, temp_path, statement_df):
        self.stmt_df = PdfUtils.get_deep_copy(statement_df)
        self.pdf_filepath = pdf_filepath
        self.temp_path = temp_path
        
    def calculate_indentation_with_postcript(self):
        # Lets process each page and see where is indentation
        # Get distinct values from 'Page' column
        distinct_pages = self.stmt_df['Page'].unique()
        filename, ext = os.path.splitext(self.pdf_filepath)
        
        for page in distinct_pages:
            
            output_text_path = os.path.join(self.temp_path, f"{filename}-{page}-gs.txt")

            lines = PdfUtils.get_pdf_page_text_data_using_gs(self.pdf_filepath, output_text_path, page)
            df_page = self.stmt_df[self.stmt_df['Page'] == page]
            print(df_page.head(10))
            
            curr_index = 0
            # Iterate through each row and check if 'Items' is of a certain type (e.g., string)
            for index, row in df_page.iterrows():
                # Check the 'Items' value
                item_value = row['Items']
                indent = -1
                if item_value and item_value.strip():
                    # Check if item_value exists in lines
                    for k in range(curr_index, len(lines)):
                        line = lines[k]
            
                        if item_value in line:
                            curr_index = k
                            indent = PdfUtils.get_leading_spaces(line)
                            print(f"Found: {item_value}, spaces={indent}")
                            break
        
                    if indent == -1:
                        print(f"NOT Found: {item_value}, spaces={indent}")
                        
                self.stmt_df.at[index, 'Indent'] = indent

    def get_Item8_consolidated_stmt_of_cfs_groups(self):
        # Create a dictionary to store row numbers
        activities_dict = {
            'Financing activities': -1,
            'Operating activities': -1,
            'Investing activities': -1
        }

        col_count_page_onwards = PdfUtils.count_columns_onwards(self.stmt_df, 'Page')

        # Chcek if last column name is 'Page'
        if col_count_page_onwards > 0:
            filtered_df = self.stmt_df[self.stmt_df['Items'].notna(
            ) & self.stmt_df.iloc[:, 1: -1 * col_count_page_onwards].isna().all(axis=1)]
        else:
            # Filter rows where "Items" is not NaN and all other columns are NaN
            filtered_df = self.stmt_df[self.stmt_df['Items'].notna(
            ) & self.stmt_df.iloc[:, 1:].isna().all(axis=1)]

        PdfUtils.print_df(filtered_df)
        # Iterate through the DataFrame and update the dictionary with row indices
        for index, row in filtered_df.iterrows():
            item = row['Items']

            # Check against each key in activities_dict
            for activity in activities_dict:
                # Calculate the match ratio using rapidfuzz
                # match_ratio = fuzz.ratio(item, activity)

                # If match ratio is greater than or equal to threshold, store the row index
                if (activity.lower() in item.lower()) & (activities_dict[activity] == -1):
                    activities_dict[activity] = (index)
                    print(f"Found {item} with index as {index}")
                    break  # Once a match is found, break the loop to avoid multiple matches

        sorted_activities_dict = {}

        for k, v in sorted(activities_dict.items(), key=lambda item: item[1]):
            sorted_activities_dict[k] = v

        return sorted_activities_dict

    
    # Function to get the subsequent entry
    def get_activity_end_index(self, activities_dict, current_key):
        # Get the list of keys in order
        keys_list = list(activities_dict.keys())
    
        # Check if the current key exists in the dictionary
        if current_key in keys_list:
            # Find the index of the current key
            current_index = keys_list.index(current_key)
    
            # Check if there is a subsequent key
            if current_index + 1 < len(keys_list):
                next_key = keys_list[current_index + 1]
                return activities_dict[next_key] - 1
            else:
                return -1  # current_key is the last one
        else:
            print(f"{current_key} not found in activities_dict.")

    def identify_groups(self):
        # Add a new column named "Group" to the dataframe with default None values
        sorted_activities_dict = self.get_Item8_consolidated_stmt_of_cfs_groups()
        
        self.stmt_df['Group'] = np.nan
        self.stmt_df['ItemType'] = np.nan
        
        for key in sorted_activities_dict.keys():
            start_index = sorted_activities_dict[key]+1
            end_index = self.get_activity_end_index(sorted_activities_dict, key)
            if end_index == -1:
                end_index = self.stmt_df.index[-1]
            print(f"For {key}, Row start: {start_index}, end: {end_index}")
            self.stmt_df.loc[start_index:end_index, 'Group'] = key
            self.stmt_df.loc[sorted_activities_dict[key]:sorted_activities_dict[key], 'ItemType'] = 'Group'
        
        PdfUtils.print_df(self.stmt_df)
        print(sorted_activities_dict)


    # Function to check condition for each row
    def grouping_row(self, row, ignore_last_cols):
        # Check if 'Item' is non-blank and ends with ':'
        if pd.notna(row['Items']) and row['Items'] != '' and row['Items'].endswith(':'):
            print(f"==== Found a row {row['Items']}")
            print(row)
            # Check if 2nd column onwards till 3rd last column has NaN, None, or blank values
            if row.iloc[1:-1 * ignore_last_cols].isna(). all() or (row.iloc[1:-1 * ignore_last_cols] == '').all():
                print("Subseq cols are found to be null")
                return True
            else:
                print("Subseq cols are found to be NOT null")
        return False

    # Helper function to merge split items based on the updated criteria
    def merge_split_items(self):
        try:
            # Identify the dynamic columns between "Items" and "Page"
            dynamic_columns = self.stmt_df.columns[1:self.stmt_df.columns.get_loc('Page')]
            df = self.stmt_df

            merged_items = []  # List to hold the merged rows
            i = 0
        
            while i < len(df):
                # Start with the current row
                current_row = df.iloc[i]
                merged_text = current_row['Items']  # Start with the initial 'Items' text
                indent_value = current_row['Indent']  # Keep track of the Indent value
                last_row_values = current_row[dynamic_columns]  # Initial values for dynamic columns
        
                # Check if all dynamic columns are NaN and ItemType is not "Group"
                if current_row[dynamic_columns].isna().all() and current_row['ItemType'] != "Group":
                    # Continue merging while the next rows also meet the criteria
                    j = i + 1
                    while j < len(df) and df.iloc[j][dynamic_columns].isna().all() and df.iloc[j]['Indent'] == indent_value and df.iloc[j]['ItemType'] != "Group":
                        # Merge the 'Items' text with the next row's 'Items' text
                        merged_text += " " + df.iloc[j]['Items']
                        last_row_values = df.iloc[j][dynamic_columns]  # Update to the last non-NaN row values
                        j += 1
        
                    # Additional check for the following row with non-NaN but same Indent and not "Group"
                    if j < len(df) and not df.iloc[j][dynamic_columns].isna().all() and df.iloc[j]['Indent'] == indent_value and df.iloc[j]['ItemType'] != "Group":
                        merged_text += " " + df.iloc[j]['Items']
                        last_row_values = df.iloc[j][dynamic_columns]  # Use the last row's values for dynamic columns
                        j += 1
        
                    # Append the merged row with last non-NaN values in dynamic columns
                    merged_row = current_row.copy()
                    merged_row['Items'] = merged_text  # Set the merged text
                    merged_row[dynamic_columns] = last_row_values  # Set last values for dynamic columns
                    merged_items.append(merged_row)
        
                    # Skip to the next unmerged row
                    i = j
                else:
                    # If conditions aren't met, add the row as-is
                    merged_items.append(current_row)
                    i += 1
        
            # Create a new DataFrame from the merged items
            df = pd.DataFrame(merged_items).reset_index(drop=True)
        except Exception as e:
            print(e)
            return
        
        self.stmt_df = df
        return

    def identify_subgroups(self):
        # Create a dictionary to store the min and max indents page-wise
        indent_stats = {}

        self.calculate_indentation_with_postcript()

        self.merge_split_items()

        # Group by 'Page' and calculate min and max Indent for each page
        for page, group in self.stmt_df.groupby('Page'):
            min_indent = group['Indent'].min()
            max_indent = group['Indent'].max()
            indent_stats[page] = {'min': min_indent, 'max': max_indent}

        print(indent_stats)

        # Get distinct values from 'Group' column
        distinct_groups = self.stmt_df['Group'].replace('', np.nan).dropna().unique()
        print(distinct_groups) 
        self.stmt_df['SubGroup'] = ""

        col_count_page_onwards = PdfUtils.count_columns_onwards(self.stmt_df, 'Page')

        prev_page = min(indent_stats)

        for group in distinct_groups:
            if group == np.nan:
                continue
                
            print(f"\n\nGroup name is {group}")
            df_group = self.stmt_df[self.stmt_df['Group'] == group]
            
            print(f"Shape is {df_group.shape}")
            
            print(df_group.shape)
            first_index_value = df_group.index[0]
            # Initialize prevItem and prevIndent with the first row
            prevItem = df_group.iloc[0]['Items']
            prevItemLoc = 0
            prevIndent = df_group.iloc[0]['Indent']
            
            print(f"First index value is {first_index_value}")
            
            # Iterate through the DataFrame starting from the 2nd row
            for pos in range(1, len(df_group)):
                currentIndent = df_group.iloc[pos]['Indent']
                item_page = df_group.iloc[pos]['Page']

                if (item_page > prev_page):
                    print(f"\n\nNew page is encountered: prev {prev_page}, new {item_page}")
                    prev_page = item_page
                    #check indent of new page and see if it is already indented
                    if currentIndent > indent_stats[item_page]['min']:
                        prevIndent = 0 #continue using prev item
                    else: #readjust
                        prevIndent = currentIndent
                
                if currentIndent == prevIndent:
                    # Update prevItem and prevIndent with the current row's values
                    prevItem = df_group.iloc[pos]['Items']
                    prevItemLoc = pos
                    prevIndent = currentIndent
                    print(f"Updated prevItem to '{prevItem}' and prevIndent to {prevIndent}")
                elif currentIndent > prevIndent:
                    # Keep iterating without updating
                    print(f"Processing row {first_index_value + pos}, Indent {currentIndent} > prevIndent {prevIndent}, prevItem is {prevItem}")
                            
                    self.stmt_df.at[first_index_value + pos, 'SubGroup'] = prevItem   
                    if pd.isna(df_group.iloc[pos]['ItemType']):
                        self.stmt_df.at[first_index_value + prevItemLoc, 'ItemType'] = 'SubGroup_1'
                else:
                    break #somehow indent has decreased
                    
            #Process it for cases where subgroup is not clear using indents   
            df_group = self.stmt_df[self.stmt_df['Group'] == group]

            groupname = None
            groupindex = -1
            # Filter rows where 'Group' is blank (either empty string or NaN)
            blank_subgroup_df = df_group[df_group['SubGroup'].isna() | (df_group['SubGroup'] == '')]
            for bs_index, row in blank_subgroup_df.iterrows():
                if self.grouping_row(row, col_count_page_onwards): 
                    groupname = row['Items']
                    groupindex = bs_index
                    self.stmt_df.at[groupindex, 'ItemType'] = 'SubGroup_1'

                    print(f"Found a group with name: {groupname} at index {groupindex}")
                    continue
                elif groupindex == -1:
                    continue
                else:
                    if bs_index == groupindex + 1:
                        groupindex = bs_index
                        print(f"Setting value at index {bs_index} and Item {row['Items']} with {groupname}")
                        self.stmt_df.at[bs_index, 'SubGroup'] = groupname
                    else: #there is a gap which shouldnt be there
                        groupindex = -1
                        groupname = None

    def identify_subgroups_level_2(self):
        # Get distinct values from 'Group' column
        #distinct_groups = self.stmt_df['Group'].replace('', np.nan).dropna().unique()
        #print(distinct_groups) 
        self.stmt_df['SubGroup_2'] = ""

        distinct_groups = self.stmt_df['Group'].replace('', np.nan).dropna().unique()
        for group in distinct_groups:
            if group == np.nan:
                continue
                
            print(f"\n\nGroup name is {group}")
            df_group_items = self.stmt_df[self.stmt_df['Group'] == group]
            distinct_sub_groups = df_group_items['SubGroup'].replace('', np.nan).dropna().unique()
            for subgroup in distinct_sub_groups:
                if subgroup == np.nan:
                    continue
            
                df_group = self.stmt_df[(self.stmt_df['Group'] == group) & (self.stmt_df['SubGroup'] == subgroup)]

                print(f"\nGroup: {group}, SubGroup: {subgroup}")
                print(df_group)  # DataFrame for each Group and SubGroup
    
                #Iterate through rows in each group_df and if you find a subsequent row that 
                # has higher Indent value then that means it is a SubGroup_2
                
                first_index_value = df_group.index[0]
                # Initialize prevItem and prevIndent with the first row
                prevItem = df_group.iloc[0]['Items']
                prevItemLoc = 0
                prevIndent = df_group.iloc[0]['Indent']
                prev_page = df_group.iloc[0]['Page']
                
                print(f"First index value is {first_index_value}")
                
                # Iterate through the DataFrame starting from the 2nd row
                for pos in range(1, len(df_group)):
                    currentIndent = df_group.iloc[pos]['Indent']
                    item_page = df_group.iloc[pos]['Page']
    
                    if (item_page > prev_page):
                        print(f"\n\nNew page is encountered: prev {prev_page}, new {item_page}")
                        print("Edge case here: difficult to figure out if this is a Level 2 subgroup or Level 1")
                        
                        min_indent = df_group['Indent'].min()
                        max_indent = df_group['Indent'].max()
                        
                        prev_page = item_page
                        #check indent of new page and see if it is already indented
                        if currentIndent > min_indent and min_indent < max_indent:
                            prevIndent = 0 #continue using prev item
                        elif min_indent < max_indent and currentIndent == min_indent: #readjust
                            prevIndent = currentIndent
                        break
                    
                    if currentIndent == prevIndent:
                        # Update prevItem and prevIndent with the current row's values
                        prevItem = df_group.iloc[pos]['Items']
                        prevItemLoc = pos
                        prevIndent = currentIndent
                        print(f"Updated prevItem to '{prevItem}' and prevIndent to {prevIndent}")
                    elif currentIndent > prevIndent:
                        # Keep iterating without updating
                        print(f"Processing row {first_index_value + pos}, Indent {currentIndent} > prevIndent {prevIndent}, prevItem is {prevItem}")
                                
                        self.stmt_df.at[first_index_value + pos, 'SubGroup_2'] = prevItem   
                        if pd.isna(df_group.iloc[pos]['ItemType']):
                            self.stmt_df.at[first_index_value + prevItemLoc, 'ItemType'] = 'SubGroup_2'
                    else:
                        break #somehow indent has decreased

    def identify_groups_and_subgroups(self):
        self.identify_groups()
        self.identify_subgroups()
        self.identify_subgroups_level_2()

        return self.stmt_df


class Mappings:   
    # Dictionary for standardized mapping of "Consolidated Statements of Cash flows"
    MAPPINGS_FOR_CONSOLIDATED_STMT_OF_CASHFLOWS = {
        "cash paid for interest": "Cash Interest",
        "Cash (refunded) paid for income taxes": "Cash Taxes",
        "Changes in operating assets and liabilities": "Change in Working Capital",
        "Net cash provided by operating activities": "Cash Flow from Operations",

        "Proceeds from borrowings of long-term debt, net of discount": "Proceeds from Debt",
        "Principal repayments of long-term debt": "Scheduled Amoritizations of Payments",
        "Proceeds from government assistance allocated to property and equipment": "Other uses of cash",

        "Cash paid for acquisition, net of cash acquired": "Acquisitions (Negative)",
        "Purchases of property and equipment": "Capital Expenditures",
        "Payments to MLSH 1 and 2 pursuant to the Tax Receivable Agreement": "Proceeds from Debt",
        "Distributions to non-controlling interests holders": "Dividends (negative)"
    }
    
    @staticmethod
    def get_consolidated_statement_of_cashflows_mappings():
        return Mappings.MAPPINGS_FOR_CONSOLIDATED_STMT_OF_CASHFLOWS
    
    
class MappingsProcessor:
    def __init__(self):
        print("MappingProcessor instantiated")
        
    # Function to preprocess text (remove special characters, stop words, and lowercasing)
    def preprocess_text(self, text):
        try:
            if isinstance(text, str):
                stop_words = set(stopwords.words('english'))
                # Remove parentheses and special characters
                text = ''.join(
                    [char for char in text if char.isalnum() or char.isspace()]).lower()
                return ' '.join([word for word in text.split() if word not in stop_words])
        except Exception as e:
            # Handle specific exception
            print(f"error occurred inside preprocess_text: {e}")
    
        return ""
    
    # Function to standardize line items with exact matching from provided standardized mapping
    
    
    def exact_match(self, line_item, mapping):
        try:
            normalized_item = self.preprocess_text(line_item)
            for key in mapping.keys():
                if self.preprocess_text(key) == normalized_item:
                    return mapping[key]
        except Exception as e:
            # Handle specific exception
            print(f"error occurred inside exact_match ({line_item}): {e}")
    
        return None  # Return None if no exact match is found
    
    
    def tfidf_cosine_similarity(self, line_items, standardized_items, confidence_threshold=0.75):
        """
        Computes the cosine similarity between line items and standardized items using TF-IDF vectorization.
    
        Parameters:
        - line_items: list of line items (strings) to standardize.
        - standardized_items: list of standardized terms (strings).
        - confidence_threshold: the minimum similarity score required to consider a match valid (0 to 1).
    
        Returns:
        - best_matches_with_scores: A list of tuples where each tuple contains the best match (standardized item)
          and its cosine similarity score for each line item. If the best match does not meet the confidence
          threshold, 'Unknown' is returned.
        """
    
        # Combine the line items and standardized terms into a single list for vectorization
        all_items = standardized_items + line_items
        tfidf_vectorizer = TfidfVectorizer()
    
        # Preprocess and vectorize the items (after cleaning the text)
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            [self.preprocess_text(item) for item in all_items])
    
        # Compute the cosine similarity between line items and standardized items
        # Rows correspond to line items, columns correspond to standardized items
        cosine_sim_matrix = cosine_similarity(
            tfidf_matrix[len(standardized_items):], tfidf_matrix[:len(standardized_items)])
    
        best_matches_with_scores = []
    
        # For each line item, find the best match based on cosine similarity
        for similarity_scores in cosine_sim_matrix:
            # Find the index of the best match (highest cosine similarity)
            best_match_idx = np.argmax(similarity_scores)
            best_score = similarity_scores[best_match_idx]
            best_match = standardized_items[best_match_idx]
    
            # Check if the best score exceeds the confidence threshold
            if best_score >= confidence_threshold:
                # If the confidence score is above the threshold, return the best match and score
                best_matches_with_scores.append((best_match, best_score))
            else:
                # If below the threshold, return 'Unknown' or some default value
                best_matches_with_scores.append(("Unknown", best_score))
    
        return best_matches_with_scores
    
    
    # Function to find the best match using fuzzy matching
    def fuzzy_map_to_standardized_value(self, text, standardized_mapping, threshold=0.80):
        # Find the best match above the threshold
        best_match = process.extractOne(
            text, standardized_mapping.keys(), scorer=fuzz.token_sort_ratio)
    
        # If the match score is above the threshold
        if best_match and best_match[1] >= threshold:
            # Return the mapped standardized value
            return standardized_mapping[best_match[0]]
    
        return None  # Return None if no good match is found
    
    
    def standardize_line_item(self, line_item, standardized_mapping, fallback_standardized_items, fuzzy_search_threshold=0.85, confidence_threshold=0.75):
        try:
            # Try to find an exact match first
    
            exact_result = self.exact_match(line_item, standardized_mapping)
            if exact_result:
                print("Match found using exact_match: " + exact_result)
                return exact_result
    
            fuzzy_result = self.fuzzy_map_to_standardized_value(
                line_item, standardized_mapping, fuzzy_search_threshold * 100)
            if fuzzy_result:
                print("Match found using fuzzy search: " + fuzzy_result)
                return fuzzy_result
    
            # If no exact match, use the updated TF-IDF based matching with confidence scores
            line_items = [line_item]
    
            # Forward: Compare line item with the values in the standardized mapping
            forward_matches = self.tfidf_cosine_similarity(
                line_items, fallback_standardized_items, confidence_threshold)
    
            # Reverse: Compare line item with the keys in the standardized mapping
            reverse_matches = self.tfidf_cosine_similarity(line_items, list(
                standardized_mapping.keys()), confidence_threshold)
    
            # Retrieve the best reverse match's corresponding value from the mapping
            reverse_match_value = standardized_mapping.get(
                reverse_matches[0][0], "Unknown")
    
            # Compare forward and reverse match scores and return the one with the higher confidence
            forward_match, forward_score = forward_matches[0]
            reverse_match, reverse_score = reverse_match_value, reverse_matches[0][1]
    
            # Return the match with the highest confidence score
            if forward_score >= reverse_score and forward_score >= confidence_threshold:
                print("Match found using cosine similarity search: " + forward_match)
                return forward_match
            elif reverse_score >= forward_score and reverse_score >= confidence_threshold:
                print("Match found using cosine similarity search: " + reverse_match)
                return reverse_match
        except Exception as e:
            # Handle specific exception
            print(f"error occurred inside exact_match ({line_item}): {e}")
    
        return ""  # Default when both are below the threshold
    
    # Function to standardize balance sheet using a combination of exact match and NLP with reverse lookup
    
    
    def standardize_statement(self, file_path, standardized_mapping, fuzzy_threshold=0.80, cosine_threshold=0.80):
        df = pd.read_csv(file_path)  # For Excel use pd.read_excel(file_path)
    
        # Create a list of fallback standardized terms (use the values of your mapping)
        fallback_standardized_items = list(standardized_mapping.values())
    
        # Apply standardization (try exact match first, then NLP, both forward and reverse)
        df['Standardized Line Item'] = df['Items'].apply(
            lambda item: self.standardize_line_item(
                item, standardized_mapping, fallback_standardized_items, fuzzy_threshold, cosine_threshold)
        )
    
        return df
    
    def standardize_consolidated_stmt_of_cfs(self, csv_file_path, fuzzy_threshold=0.80, cosine_threshold=0.80):
        return self.standardize_statement(csv_file_path, Mappings.get_consolidated_statement_of_cashflows_mappings(), fuzzy_threshold, cosine_threshold)


class FinancialStmtProcessor:
    
    # Function to extract year or month-year
    def extract_year_or_month_year(self, value):
        # Check if value is a 4-digit year starting with '20'
        if value.isdigit() and 2000 <= int(value) <= 2099:
            return value  # Use the year as column name
    
        # Attempt to parse date if value is not a year
        try:
            date = datetime.strptime(value, '%b %d, %Y')  # e.g., "Dec 31, 2023"
        except ValueError:
            try:
                date = datetime.strptime(value, '%B %d, %Y')  # e.g., "December 31, 2023"
            except ValueError:
                return None  # Not a valid year or date format
        return f"{date.strftime('%b')} {date.year}"  # Format as "Month Year"
    


    def clean_financial_columns(self, dataframe):
        start_idx = dataframe.columns.get_loc('Items') + 1
        dynamic_columns = dataframe.columns[start_idx:dataframe.columns.get_loc('Page')]
        
        col_names = {}
        # Check if the first row has 4-digit numbers starting with "20" in these columns
        for col in dynamic_columns:
            first_row_value = str(dataframe.at[0, col])  # Convert to string for consistency
            new_name = self.extract_year_or_month_year(first_row_value)
            if new_name:
                col_names[col] = new_name
            else:
                col_names.clear()
                break
                
        if len(col_names) > 0:
            for col in col_names:
                dataframe.rename(columns={col: col_names[col]}, inplace=True)

        # re-read the cols in case they get updated above
        dynamic_columns = dataframe.columns[start_idx:dataframe.columns.get_loc('Page')]

        # Iterate through each dynamic column and apply transformations
        for col in dynamic_columns:
            # Remove any extraneous spaces and commas
            dataframe[col] = dataframe[col].str.strip().replace(',', '', regex=True)
            dataframe[col] = dataframe[col].replace('[\$,]', '', regex=True)  # Remove commas and dollar signs
            
            # Convert numbers in (1234), (1,234), (1234.56), (1,234.56) format to -1234, -1234, -1234.56, -1234.56
            dataframe[col] = dataframe[col].replace(r'\(([\d,]+\.\d+|\d+)\)', r'-\1', regex=True)
            
            # Convert to numeric (handles cases where values are still non-numeric or NaN)
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        
        return dataframe


    def __init__(self, doc_type: DocumentType, inpath, outpath, filename):
        self.pdf_reader = None
        
        if isinstance(doc_type, DocumentType):  # Check if the input is of enum type
            self.doc_type = doc_type  
        else:
            raise ValueError("Invalid doc type passed")
            
        print("Instantitated FinancialStmtProcessor for doc type {doc_type.name}")
        
        if(doc_type == DocumentType.TEN_K):
            self.pdf_reader = PdfTableReader(inpath, outpath, filename)
            
    def run(self):
        if((self.doc_type == DocumentType.TEN_K) & isinstance(self.pdf_reader, PdfTableReader)):          
            # Extract Consolidated Stmt of CFs (includes clean and merge)
            statement_df = self.pdf_reader.get_consolidated_statement_of_cashflows()

            # Group/Subgroup the statement
            stmt_group_processor = StatementGroupProcessor(
                self.pdf_reader.get_input_filename(), 
                self.pdf_reader.get_temp_path(),
                statement_df)
        
            processed_df_groups_subgroups = stmt_group_processor.identify_groups_and_subgroups()
            
            # Clean the numbers
            processed_df = self.clean_financial_columns(processed_df_groups_subgroups)
        
            PdfUtils.save_df_as_csv(processed_df, self.pdf_reader.get_temp_csv_filepathname())

            mappings_processor = MappingsProcessor()
            standard_df = mappings_processor.standardize_consolidated_stmt_of_cfs(
                self.pdf_reader.get_temp_csv_filepathname(), 
                fuzzy_threshold=0.85, 
                cosine_threshold=0.80)
            
            # Clean and Save the DF
            standard_df['Standardized Line Item'] = standard_df['Standardized Line Item'].replace(
                r'^\s*$', np.nan, regex=True)
            df_cleaned = standard_df.dropna(subset=['Standardized Line Item'])
            PdfUtils.print_df(df_cleaned)
              
            PdfUtils.save_df_as_csv(standard_df, self.pdf_reader.get_standard_csv_filepathname())
            return f"{self.pdf_reader.get_standard_csv_filepathname()}"
            
# In[]

output_files = {}

def process_file(filename, input_dir, output_dir, doc_type):
    try:
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Error: File '{input_path}' does not exist.")
            return
    
        os.makedirs(output_dir, exist_ok=True)
        
        doctype_enum = DocumentType.from_string(doc_type)
        
        if not doctype_enum:
            print("Document type is invalid. Supported doc types are 10-K, 10-Q or HF")
    
        processor = FinancialStmtProcessor(doctype_enum, input_dir, output_dir, filename)
        output_filepath = processor.run()
    
        print(f"Processed file saved to: {output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"Failed to process file: {filename}")

    return None

def main(args=None):
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Name of the file to process")
    parser.add_argument("-i", "--input_dir", type=str, default="./in", help="Directory where the input file is located")
    parser.add_argument("-o", "--output_dir", type=str, default="./out", help="Directory where the processed file will be saved")
    parser.add_argument("-t", "--type", type=str, choices=["10-K", "10-Q", "HF"], required=True, help="Type of data in the file (e.g., '10K' or '10Q')")

    if args is None:
        # Default values for debugging in IDE
        #args = argparse.Namespace(file="Maravai.pdf", input_dir="./in", output_dir="./out", type="10-K")
        
        args_list = [
            ["-f", "newell.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "unfi.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],

"""
            ["-f", "Dana.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],

            ["-f", "Maravai.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "att.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "CCC.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "GrafTech.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "GlobalNetLease.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "KraftHeinz.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "BellRing.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
            ["-f", "PostHoldings.pdf", "-i", "./in", "-o", "./out", "-t", "10-K"],
"""
        ]
        
        for arg in args_list:
            # Parse arguments and convert them into a dictionary
            parsed_args = parser.parse_args(arg)
            args_dict = vars(parsed_args)  # Convert Namespace to dictionary
            # Create a new Namespace object from the dictionary
            args = argparse.Namespace(**args_dict)
            # Call the processing function with arguments
            output_filepath = process_file(args.file, args.input_dir, args.output_dir, args.type)
            if output_filepath is None:
                output_files[args.file] = output_filepath
                #output_files.append(output_filepath)
                print(f"Failed to process file: {args.file}")
            else:
                output_files[args.file] = output_filepath
                print(f"Processed file: {output_filepath}")           
    else:
        args = parser.parse_args(args)

        # Call the processing function with arguments
        output_filepath = process_file(args.file, args.input_dir, args.output_dir, args.type)
        print(output_filepath)
    
    print("Done with processing")
    
if __name__ == "__main__":
    main()
    print("Done with main")
    
# In[]
dfs = []

expected_rows = {}
expected_rows['GlobalNetLease.pdf'] = 70
expected_rows['KraftHeinz.pdf'] = 48
expected_rows['BellRing.pdf'] = 34
expected_rows['Maravai.pdf'] = 58
expected_rows['PostHoldings.pdf'] = 59
expected_rows['GrafTech.pdf'] = 48
expected_rows['CCC.pdf'] = 71
expected_rows['att.pdf'] = 53
expected_rows['Dana.pdf'] = 52
expected_rows['newell.pdf'] = 52
expected_rows['unfi.pdf'] = 52


for key, value in output_files.items():
    print(f"File: {key}, Output path: {value}")
    
for key, value in output_files.items():
    print(f"Displaying CSV filedata: {value}")
    if(value is None):
        print(f"ERROR: Failed to process file {key}")
        continue
    
    df = pd.read_csv(value)
    dfs.append(df)
    try:
        if(df.shape[0] != expected_rows[key]):
            print(f"{key}: MISMATCH!!!: The number of rows in dataframe {df.shape[0]} vs expected {expected_rows[key]}")
    except Exception as e:
        # Handle specific exception
        print(f"error occurred: {e}")
    #PdfUtils.print_df(df)

# In[]
def get_final_output(df):
    filtered_df = df[df['Standardized Line Item'].notna() & (df['Standardized Line Item'] != '')].copy()
    
    # Iterate through the filtered DataFrame row by row
    for idx, row in filtered_df.iterrows():
        # Check if ItemType is 'SubGroup_1' or 'SubGroup_2'
        if row['ItemType'] in ['SubGroup_1', 'SubGroup_2']:
            # Get the 'Items' and 'Group' values from the current row
            items_value = row['Items']
            group_value = row['Group']
            colname = 'SubGroup'
            if(row['ItemType'] == 'SubGroup_2'):
                colname ='SubGroup_2'
                
            # Sum all the dynamic columns in the original DataFrame where Group and Items match
            dynamic_columns = df.columns[1:df.columns.get_loc('Page')]
            matching_rows = df[(df['Group'] == group_value) & (df[colname] == items_value)]
            sum_values = matching_rows[dynamic_columns].sum()
    
            for col in dynamic_columns:
                filtered_df.at[idx, col] = sum_values[col]
            
            # Output the result for the current row
            # print(f"Sum of dynamic columns for Group='{group_value}', Items='{items_value}':\n{sum_values}\n")

    filtered_df = filtered_df.drop(['Page', 'Indent', 'SubGroup', 'SubGroup_2'], axis=1)

    return filtered_df

# In[]
def generate_final_datafiles():
    filtered_dfs = []
    for filename in output_files.values():
        print(f"\n\n\nDisplaying CSV filedata: {filename}")
        df = pd.read_csv(filename)
        filtered_df = get_final_output(df)
        filtered_df.shape
    
        filtered_dfs.append(filtered_df)
        PdfUtils.print_df(filtered_df)
    
        # Split the filename into parts before and after the last underscore
        base, ext = os.path.splitext(filename)  # Separate base and extension
        parts = base.rsplit('_', 1)  # Split at the last underscore
        
        # Insert "_final_" and reassemble the filename
        new_filename = f"{parts[0]}_final_{parts[1]}{ext}"
    
        PdfUtils.save_df_as_csv(filtered_df, new_filename)

generate_final_datafiles()

# In[]
print("Done")

# Filtered copy of DataFrame where 'Standardized Line Item' is non-blank and non-NA
