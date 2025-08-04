I have a balance sheet, I want to write a prompt that look at each balance sheet entry one by one and if they belong to one of the given categories, returns that standardize category name. The objective is to standardize balance sheet from different companies into a standardized one with consistent item names. Write a mult-shot prompt that is in a json file and loaded by the python script to be fed into Open AI 3.5. Also, the line items will be fed one by one and the line item and response should be appended to the prompt before sending next line item to Generative AI. The balance sheet table is extracted in a python dataframe and the first column is 'Line Item'. Enhance the prompt by giving some extra content for each standardized category, with comprehensive category definitions. Each balance sheet has two main groups - Assets and Liabilities. Process one group at a time instead of entire balance sheet dataframe. Also see if the response can also include reasoning as well that we can save in a text file for analysis and debugging purposes later?

```json
{
  "system": "You are a financial analyst AI that specializes in standardizing balance sheet line items from various formats into a consistent chart of accounts. You will receive one balance sheet line item at a time, and your task is to return the best-matching standardized category name along with a short reasoning for your classification. Respond in JSON format: {\"standard_category\": \"...\", \"reasoning\": \"...\"}. Only use the categories defined below. Do not invent new ones.",
  "category_definitions": {
    "Assets": {
      "Cash and Cash Equivalents": "Highly liquid assets such as physical cash, bank deposits, money market funds, and short-term government bonds.",
      "Accounts Receivable": "Money owed to the company by customers for goods or services delivered.",
      "Inventory": "Raw materials, work-in-progress, and finished goods held for sale.",
      "Prepaid Expenses": "Payments made in advance for goods/services to be received in the future (e.g., insurance, rent).",
      "Property, Plant, and Equipment (PP&E)": "Tangible fixed assets including buildings, machinery, and land.",
      "Intangible Assets": "Non-physical assets like patents, trademarks, goodwill, and software.",
      "Investments": "Long-term investments including equities, bonds, and joint ventures not classified as cash equivalents.",
      "Other Assets": "Residual category for assets that don’t fit other definitions clearly (e.g., deferred tax assets, security deposits)."
    },
    "Liabilities": {
      "Accounts Payable": "Obligations to suppliers for goods and services received but not yet paid for.",
      "Accrued Expenses": "Liabilities for expenses that have been incurred but not yet paid (e.g., wages, utilities).",
      "Short-term Debt": "Borrowings due within one year (e.g., lines of credit, bank loans).",
      "Long-term Debt": "Borrowings due beyond one year, including bonds and long-term loans.",
      "Deferred Revenue": "Advance payments received for products/services not yet delivered.",
      "Lease Liabilities": "Present value of future lease payments under IFRS 16 or ASC 842.",
      "Other Liabilities": "Catch-all for liabilities that don’t clearly fit into other categories (e.g., deferred tax liabilities, asset retirement obligations)."
    }
  },
  "examples": [
    {
      "input": "Cash in bank",
      "output": {
        "standard_category": "Cash and Cash Equivalents",
        "reasoning": "Cash in bank is a liquid asset, readily available for use, fitting the definition of cash equivalents."
      }
    },
    {
      "input": "Trade receivables",
      "output": {
        "standard_category": "Accounts Receivable",
        "reasoning": "Trade receivables represent money owed by customers, aligning with Accounts Receivable."
      }
    },
    {
      "input": "Machinery and Equipment",
      "output": {
        "standard_category": "Property, Plant, and Equipment (PP&E)",
        "reasoning": "Machinery and equipment are tangible fixed assets used in operations."
      }
    },
    {
      "input": "Deferred tax liability",
      "output": {
        "standard_category": "Other Liabilities",
        "reasoning": "Deferred tax liabilities don't fit into operational liabilities and are long-term obligations."
      }
    }
  ]
}
```


```py
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Multi-shot prompt: System message + 3 examples + user input
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful legal assistant. Your job is to read legal clauses "
            "and explain them in plain English that non-lawyers can understand. "
            "Avoid legal jargon and keep the explanations clear and simple."
        )
    },
    
    # Example 1
    {
        "role": "user",
        "content": (
            "Summarize the following legal clause in plain English:\n\n"
            "\"This Agreement shall be governed by and construed in accordance with the laws of the State of California.\""
        )
    },
    {
        "role": "assistant",
        "content": "This agreement will follow the laws of California."
    },

    # Example 2
    {
        "role": "user",
        "content": (
            "Summarize the following legal clause in plain English:\n\n"
            "\"The failure of either party to enforce any provision of this Agreement shall not be construed as a waiver of such provision.\""
        )
    },
    {
        "role": "assistant",
        "content": "If one side doesn’t enforce a rule in the contract, it doesn’t mean they give up the right to do it later."
    },

    # Example 3
    {
        "role": "user",
        "content": (
            "Summarize the following legal clause in plain English:\n\n"
            "\"Each party shall keep the terms and conditions of this Agreement confidential and shall not disclose any such information to third parties.\""
        )
    },
    {
        "role": "assistant",
        "content": "Both sides must keep the agreement private and not share its details with others."
    },

    # Actual user prompt
    {
        "role": "user",
        "content": (
            "Summarize the following legal clause in plain English:\n\n"
            "\"Notwithstanding anything to the contrary contained herein, either party may terminate this Agreement "
            "upon thirty (30) days prior written notice to the other party.\""
        )
    }
]

# Generate response from OpenAI Chat API
response = openai.ChatCompletion.create(
    model="gpt-4",  # or "gpt-3.5-turbo"
    messages=messages,
    temperature=0.3
)

# Print the model's response
print("Model output:\n", response['choices'][0]['message']['content'])
```
