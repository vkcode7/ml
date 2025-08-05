import json
import pandas as pd
from typing import Dict, Any, List

class BalanceSheetStandardizer:
    def __init__(self, prompt_file_path: str):
        """
        Initialize the balance sheet standardizer with the JSON prompt file.
        
        Args:
            prompt_file_path (str): Path to the JSON file containing the prompt configuration
        """
        with open(prompt_file_path, 'r') as f:
            self.prompt_config = json.load(f)
        
        # Initialize conversation history for context building
        self.conversation_history = []
    
    def build_prompt(self, line_item: str) -> str:
        """
        Build the complete prompt including system message, examples, and current line item.
        
        Args:
            line_item (str): The balance sheet line item to classify
            
        Returns:
            str: Complete prompt ready to send to AI
        """
        # Start with system prompt
        prompt_parts = [self.prompt_config["system_prompt"]]
        
        # Add standardized categories with detailed information
        prompt_parts.append("\nSTANDARDIZED CATEGORIES WITH DEFINITIONS:")
        for category, details in self.prompt_config["standardized_categories"].items():
            prompt_parts.append(f"\n{category}:")
            prompt_parts.append(f"  Description: {details['description']}")
            prompt_parts.append(f"  Common variations: {', '.join(details['common_variations'])}")
            if details.get('includes'):
                prompt_parts.append(f"  Includes: {', '.join(details['includes'])}")
            if details.get('excludes'):
                prompt_parts.append(f"  Excludes: {', '.join(details['excludes'])}")
        
        # Add few-shot examples
        prompt_parts.append("\nEXAMPLES:")
        for example in self.prompt_config["few_shot_examples"]:
            prompt_parts.append(f"Line Item: {example['line_item']}")
            prompt_parts.append(f"Standardized Category: {example['standardized_category']}")
            prompt_parts.append("")
        
        # Add conversation history (previous line items and responses)
        if self.conversation_history:
            prompt_parts.append("PREVIOUS CLASSIFICATIONS:")
            for item in self.conversation_history:
                prompt_parts.append(f"Line Item: {item['line_item']}")
                prompt_parts.append(f"Standardized Category: {item['response']}")
                prompt_parts.append("")
        
        # Add current instruction
        current_instruction = self.prompt_config["instruction_template"].format(line_item=line_item)
        prompt_parts.append(current_instruction)
        
        # Add response format reminder
        prompt_parts.append(f"\n{self.prompt_config['response_format']}")
        
        return "\n".join(prompt_parts)
    
    def add_to_history(self, line_item: str, response: str):
        """
        Add the processed line item and response to conversation history.
        
        Args:
            line_item (str): The processed line item
            response (str): The AI's response (standardized category)
        """
        self.conversation_history.append({
            "line_item": line_item,
            "response": response.strip()
        })
    
    def identify_balance_sheet_groups(self, df: pd.DataFrame, group_column: str = None) -> Dict[str, pd.DataFrame]:
        """
        Identify and separate Assets from Liabilities & Equity groups in the balance sheet.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Line Item' column
            group_column (str): Optional column name that explicitly identifies groups
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with 'ASSETS' and 'LIABILITIES_EQUITY' DataFrames
        """
        if group_column and group_column in df.columns:
            # Use explicit group column if provided
            assets_df = df[df[group_column].str.contains('asset', case=False, na=False)].copy()
            liab_equity_df = df[~df[group_column].str.contains('asset', case=False, na=False)].copy()
        else:
            # Use heuristics to identify groups based on line item names and patterns
            line_items = df['Line Item'].str.lower()
            
            # Find potential split points
            total_assets_idx = None
            for idx, item in enumerate(line_items):
                if 'total asset' in item or item == 'total assets':
                    total_assets_idx = idx
                    break
            
            if total_assets_idx is not None:
                # Split at total assets
                assets_df = df.iloc[:total_assets_idx + 1].copy()
                liab_equity_df = df.iloc[total_assets_idx + 1:].copy()
            else:
                # Fallback: try to identify by common patterns
                asset_keywords = ['cash', 'receivable', 'inventory', 'prepaid', 'property', 'plant', 'equipment', 
                                'goodwill', 'intangible', 'investment', 'asset']
                liability_keywords = ['payable', 'debt', 'loan', 'accrued', 'deferred', 'liability', 
                                    'stock', 'equity', 'earnings', 'retained']
                
                # Simple heuristic: items with asset keywords go to assets, others to liabilities
                asset_mask = line_items.str.contains('|'.join(asset_keywords), case=False, na=False)
                
                assets_df = df[asset_mask].copy()
                liab_equity_df = df[~asset_mask].copy()
        
        return {
            'ASSETS': assets_df,
            'LIABILITIES_EQUITY': liab_equity_df
        }
    
    def build_group_prompt(self, group_items: List[str], group_type: str) -> str:
        """
        Build prompt for processing a specific balance sheet group.
        
        Args:
            group_items (List[str]): Line items for this group
            group_type (str): Either 'ASSETS' or 'LIABILITIES_EQUITY'
            
        Returns:
            str: Complete prompt for group processing
        """
        # Start with system prompt
        prompt_parts = [self.prompt_config["system_prompt"]]
        
        # Determine relevant categories for this group
        if group_type == 'ASSETS':
            relevant_categories = self.prompt_config["assets_categories"]
            group_display_name = "Assets"
        else:
            relevant_categories = self.prompt_config["liabilities_equity_categories"] 
            group_display_name = "Liabilities & Equity"
        
        # Add relevant categories with detailed information
        prompt_parts.append(f"\nRELEVANT {group_display_name.upper()} CATEGORIES:")
        for category in relevant_categories:
            if category in self.prompt_config["standardized_categories"]:
                details = self.prompt_config["standardized_categories"][category]
                prompt_parts.append(f"\n{category}:")
                prompt_parts.append(f"  Description: {details['description']}")
                prompt_parts.append(f"  Common variations: {', '.join(details['common_variations'])}")
                if details.get('includes'):
                    prompt_parts.append(f"  Includes: {', '.join(details['includes'])}")
                if details.get('excludes'):
                    prompt_parts.append(f"  Excludes: {', '.join(details['excludes'])}")
        
        # Add relevant few-shot examples
        prompt_parts.append(f"\n{group_display_name.upper()} EXAMPLES:")
        for example in self.prompt_config["few_shot_examples"]:
            if example['standardized_category'] in relevant_categories:
                prompt_parts.append(f"Line Item: {example['line_item']}")
                prompt_parts.append(f"Standardized Category: {example['standardized_category']}")
                prompt_parts.append("")
        
        # Format line items
        formatted_items = "\n".join([f"- {item}" for item in group_items])
        
        # Add group instruction
        group_instruction = self.prompt_config["group_instruction_template"].format(
            group_type=group_display_name,
            group_items=formatted_items
        )
        prompt_parts.append(group_instruction)
        
        # Add response format
        response_format = self.prompt_config["group_response_format"].format(
            group_type=group_display_name
        )
        prompt_parts.append(f"\n{response_format}")
        
        return "\n".join(prompt_parts)
    
    def process_balance_sheet_by_groups(self, df: pd.DataFrame, ai_client_function, group_column: str = None) -> tuple:
        """
        Process balance sheet by groups (Assets, then Liabilities & Equity) with detailed reasoning.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Line Item' column
            ai_client_function: Function that takes a prompt and returns AI response
            group_column (str): Optional column that explicitly identifies groups
            
        Returns:
            tuple: (result_df, reasoning_data) where reasoning_data contains detailed analysis
        """
        # Identify balance sheet groups
        groups = self.identify_balance_sheet_groups(df, group_column)
        
        # Initialize results
        all_mappings = {}
        all_reasoning = {}
        
        # Process each group
        for group_type, group_df in groups.items():
            if len(group_df) == 0:
                continue
                
            print(f"\nProcessing {group_type} group ({len(group_df)} items)...")
            
            # Extract line items for this group
            group_line_items = group_df['Line Item'].tolist()
            
            # Build group-specific prompt
            prompt = self.build_group_prompt(group_line_items, group_type)
            
            # Get AI response
            response = ai_client_function(prompt)
            
            # Parse JSON response with reasoning
            try:
                import json
                group_response = json.loads(response)
                
                # Extract mappings and reasoning
                for item, details in group_response.items():
                    if isinstance(details, dict) and 'category' in details:
                        # New format with reasoning
                        all_mappings[item] = details['category']
                        all_reasoning[item] = {
                            'group': group_type,
                            'category': details['category'],
                            'reasoning': details.get('reasoning', 'No reasoning provided'),
                            'confidence': details.get('confidence', 'UNKNOWN'),
                            'alternative_categories': details.get('alternative_categories', []),
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                    else:
                        # Fallback for simple format
                        all_mappings[item] = str(details)
                        all_reasoning[item] = {
                            'group': group_type,
                            'category': str(details),
                            'reasoning': 'Simple classification without detailed reasoning',
                            'confidence': 'UNKNOWN',
                            'alternative_categories': [],
                            'timestamp': pd.Timestamp.now().isoformat()
                        }
                    
                    print(f"  {item} -> {all_mappings[item]} (Confidence: {all_reasoning[item]['confidence']})")
                    
            except json.JSONDecodeError:
                print(f"Error: AI response for {group_type} is not valid JSON. Skipping group.")
                # Log the raw response for debugging
                all_reasoning[f"{group_type}_ERROR"] = {
                    'group': group_type,
                    'category': 'ERROR',
                    'reasoning': f'JSON parsing failed. Raw response: {response[:500]}...',
                    'confidence': 'ERROR',
                    'alternative_categories': [],
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                continue
        
        # Create result dataframe
        result_df = df.copy()
        standardized_categories = []
        confidence_levels = []
        
        for line_item in df['Line Item']:
            if line_item in all_mappings:
                category = self.validate_response(all_mappings[line_item])
                standardized_categories.append(category)
                confidence_levels.append(all_reasoning[line_item]['confidence'])
            else:
                print(f"Warning: No mapping found for '{line_item}', defaulting to 'OTHER'")
                standardized_categories.append("OTHER")
                confidence_levels.append("LOW")
                # Add reasoning for missing items
                all_reasoning[line_item] = {
                    'group': 'UNKNOWN',
                    'category': 'OTHER',
                    'reasoning': 'No mapping found in AI response, defaulted to OTHER',
                    'confidence': 'LOW',
                    'alternative_categories': [],
                    'timestamp': pd.Timestamp.now().isoformat()
                }
        
        result_df['Standardized_Category'] = standardized_categories
        result_df['Confidence'] = confidence_levels
        
        return result_df, all_reasoning
    
    def save_reasoning_analysis(self, reasoning_data: Dict, filepath: str):
        """
        Save detailed reasoning analysis to a text file for debugging and audit purposes.
        
        Args:
            reasoning_data (Dict): Reasoning data from processing
            filepath (str): Path to save the analysis file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("BALANCE SHEET STANDARDIZATION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            total_items = len(reasoning_data)
            confidence_counts = {}
            category_counts = {}
            
            for item, details in reasoning_data.items():
                if details['category'] != 'ERROR':
                    conf = details['confidence']
                    cat = details['category']
                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total items processed: {total_items}\n")
            f.write(f"Confidence distribution: {confidence_counts}\n")
            f.write(f"Category distribution: {dict(sorted(category_counts.items()))}\n\n")
            
            # Group analysis
            groups = {}
            for item, details in reasoning_data.items():
                group = details['group']
                if group not in groups:
                    groups[group] = []
                groups[group].append((item, details))
            
            # Detailed analysis by group
            for group, items in groups.items():
                f.write(f"\n{group} GROUP ANALYSIS\n")
                f.write("=" * (len(group) + 15) + "\n\n")
                
                for item, details in items:
                    f.write(f"Line Item: {item}\n")
                    f.write(f"Category: {details['category']}\n")
                    f.write(f"Confidence: {details['confidence']}\n")
                    f.write(f"Reasoning: {details['reasoning']}\n")
                    
                    if details['alternative_categories']:
                        f.write(f"Alternatives considered: {', '.join(details['alternative_categories'])}\n")
                    
                    f.write(f"Timestamp: {details['timestamp']}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Quality analysis
            f.write("\nQUALITY ANALYSIS\n")
            f.write("=" * 16 + "\n\n")
            
            # High confidence items
            high_conf = [item for item, details in reasoning_data.items() 
                        if details['confidence'] == 'HIGH']
            f.write(f"High confidence classifications: {len(high_conf)} items\n")
            
            # Low confidence items (potential issues)
            low_conf = [item for item, details in reasoning_data.items() 
                       if details['confidence'] == 'LOW']
            if low_conf:
                f.write(f"\nLOW CONFIDENCE ITEMS (require review):\n")
                for item in low_conf:
                    details = reasoning_data[item]
                    f.write(f"- {item}: {details['category']} ({details['reasoning'][:100]}...)\n")
            
            # Items with alternatives
            with_alternatives = [item for item, details in reasoning_data.items()
                               if details['alternative_categories']]
            if with_alternatives:
                f.write(f"\nITEMS WITH ALTERNATIVE CATEGORIES CONSIDERED:\n")
                for item in with_alternatives:
                    details = reasoning_data[item]
                    f.write(f"- {item}: Chose {details['category']} over {details['alternative_categories']}\n")
            
            # Error items
            error_items = [item for item, details in reasoning_data.items()
                          if details['category'] == 'ERROR']
            if error_items:
                f.write(f"\nERROR ITEMS (require manual review):\n")
                for item in error_items:
                    details = reasoning_data[item]
                    f.write(f"- {item}: {details['reasoning']}\n")

    def save_reasoning_json(self, reasoning_data: Dict, filepath: str):
        """
        Save reasoning data as JSON for programmatic analysis.
        
        Args:
            reasoning_data (Dict): Reasoning data from processing
            filepath (str): Path to save the JSON file
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
        """
        Process balance sheet by groups (Assets, then Liabilities & Equity).
        
        Args:
            df (pd.DataFrame): DataFrame with 'Line Item' column
            ai_client_function: Function that takes a prompt and returns AI response
            group_column (str): Optional column that explicitly identifies groups
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'Standardized_Category' column
        """
        # Identify balance sheet groups
        groups = self.identify_balance_sheet_groups(df, group_column)
        
        # Initialize results
        all_mappings = {}
        
        # Process each group
        for group_type, group_df in groups.items():
            if len(group_df) == 0:
                continue
                
            print(f"\nProcessing {group_type} group ({len(group_df)} items)...")
            
            # Extract line items for this group
            group_line_items = group_df['Line Item'].tolist()
            
            # Build group-specific prompt
            prompt = self.build_group_prompt(group_line_items, group_type)
            
            # Get AI response
            response = ai_client_function(prompt)
            
            # Parse JSON response
            try:
                import json
                group_mappings = json.loads(response)
                all_mappings.update(group_mappings)
                
                # Log mappings for this group
                for item, category in group_mappings.items():
                    print(f"  {item} -> {category}")
                    
            except json.JSONDecodeError:
                print(f"Error: AI response for {group_type} is not valid JSON. Skipping group.")
                continue
        
        # Create result dataframe
        result_df = df.copy()
        standardized_categories = []
        
        for line_item in df['Line Item']:
            if line_item in all_mappings:
                category = self.validate_response(all_mappings[line_item])
                standardized_categories.append(category)
            else:
                print(f"Warning: No mapping found for '{line_item}', defaulting to 'OTHER'")
                standardized_categories.append("OTHER")
        
        result_df['Standardized_Category'] = standardized_categories
        return result_df
        """
        Build prompt for processing entire balance sheet at once.
        
        Args:
            line_items (List[str]): All balance sheet line items
            
        Returns:
            str: Complete prompt for batch processing
        """
        # Start with system prompt
        prompt_parts = [self.prompt_config["system_prompt"]]
        
        # Add standardized categories with detailed information
        prompt_parts.append("\nSTANDARDIZED CATEGORIES WITH DEFINITIONS:")
        for category, details in self.prompt_config["standardized_categories"].items():
            prompt_parts.append(f"\n{category}:")
            prompt_parts.append(f"  Description: {details['description']}")
            prompt_parts.append(f"  Common variations: {', '.join(details['common_variations'])}")
            if details.get('includes'):
                prompt_parts.append(f"  Includes: {', '.join(details['includes'])}")
            if details.get('excludes'):
                prompt_parts.append(f"  Excludes: {', '.join(details['excludes'])}")
        
        # Add few-shot examples
        prompt_parts.append("\nEXAMPLES:")
        for example in self.prompt_config["few_shot_examples"]:
            prompt_parts.append(f"Line Item: {example['line_item']}")
            prompt_parts.append(f"Standardized Category: {example['standardized_category']}")
            prompt_parts.append("")
        
        # Format line items for batch processing
        formatted_items = "\n".join([f"- {item}" for item in line_items])
        
        # Add batch instruction
        batch_instruction = self.prompt_config["batch_instruction_template"].format(
            balance_sheet_items=formatted_items
        )
        prompt_parts.append(batch_instruction)
        
        # Add response format
        prompt_parts.append(f"\n{self.prompt_config['batch_response_format']}")
        
        return "\n".join(prompt_parts)
    
    def process_balance_sheet_batch(self, df: pd.DataFrame, ai_client_function) -> pd.DataFrame:
        """
        Process entire balance sheet at once for better accuracy.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Line Item' column
            ai_client_function: Function that takes a prompt and returns AI response
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'Standardized_Category' column
        """
        # Extract all line items
        line_items = df['Line Item'].tolist()
        
        # Build batch prompt
        prompt = self.build_batch_prompt(line_items)
        
        # Get AI response
        response = ai_client_function(prompt)
        
        # Parse JSON response
        try:
            import json
            mappings = json.loads(response)
        except json.JSONDecodeError:
            print("Error: AI response is not valid JSON. Falling back to individual processing.")
            return self.process_balance_sheet_individual(df, ai_client_function)
        
        # Create result dataframe
        result_df = df.copy()
        standardized_categories = []
        
        for line_item in line_items:
            if line_item in mappings:
                category = self.validate_response(mappings[line_item])
                standardized_categories.append(category)
                print(f"Mapped: {line_item} -> {category}")
            else:
                print(f"Warning: No mapping found for '{line_item}', defaulting to 'OTHER'")
                standardized_categories.append("OTHER")
        
        result_df['Standardized_Category'] = standardized_categories
        return result_df
        """
        Process an entire balance sheet DataFrame one item at a time.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Line Item' column
            ai_client_function: Function that takes a prompt and returns AI response
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'Standardized_Category' column
        """
        # Create a copy of the dataframe
        result_df = df.copy()
        standardized_categories = []
        
        # Process each row
        for index, row in df.iterrows():
            line_item = row['Line Item']
            
            # Build prompt for current line item
            prompt = self.build_prompt(line_item)
            
            # Get AI response (you'll need to implement this based on your AI client)
            response = ai_client_function(prompt)
            
            # Clean and validate response
            standardized_category = self.validate_response(response)
            standardized_categories.append(standardized_category)
            
            # Add to conversation history for context in next iterations
            self.add_to_history(line_item, standardized_category)
            
            print(f"Processed: {line_item} -> {standardized_category}")
        
        # Add standardized categories to dataframe
        result_df['Standardized_Category'] = standardized_categories
        return result_df
    
    def validate_response(self, response: str) -> str:
        """
        Validate that the AI response is a valid standardized category.
        
        Args:
            response (str): Raw AI response
            
        Returns:
            str: Validated standardized category
        """
        cleaned_response = response.strip().upper()
        
        # Check if response is in valid categories
        if cleaned_response in self.prompt_config["standardized_categories"]:
            return cleaned_response
        
        # If not found, return OTHER
        print(f"Warning: Invalid response '{response}', defaulting to 'OTHER'")
        return "OTHER"
    
    def save_conversation_history(self, filepath: str):
        """
        Save the conversation history to a JSON file for analysis.
        
        Args:
            filepath (str): Path to save the conversation history
        """
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

# Example usage
def enhanced_example_ai_client(prompt: str) -> str:
    """
    Enhanced placeholder for AI client that returns reasoning-rich responses.
    Replace this with your actual AI API call.
    
    Args:
        prompt (str): The prompt to send to AI
        
    Returns:
        str: AI response with detailed reasoning
    """
    # This is a more realistic example response with reasoning
    sample_response = {
        "Cash and cash equivalents": {
            "category": "CASH_AND_CASH_EQUIVALENTS",
            "reasoning": "This line item clearly represents highly liquid assets that can be readily converted to cash. The terminology 'cash equivalents' indicates short-term, highly liquid investments that are readily convertible to known amounts of cash.",
            "confidence": "HIGH",
            "alternative_categories": []
        },
        "Short-term investments": {
            "category": "SHORT_TERM_INVESTMENTS", 
            "reasoning": "These are marketable securities with maturities likely between 90 days and 1 year. Positioned in current assets section indicates short-term nature.",
            "confidence": "HIGH",
            "alternative_categories": ["CASH_AND_CASH_EQUIVALENTS"]
        },
        "Trade receivables": {
            "category": "ACCOUNTS_RECEIVABLE",
            "reasoning": "Standard terminology for amounts owed by customers for goods sold or services provided on credit. 'Trade' qualifier confirms these are operational receivables.",
            "confidence": "HIGH", 
            "alternative_categories": []
        }
    }
    
    import json
    return json.dumps(sample_response)
    """
    Placeholder for your AI client function.
    Replace this with your actual AI API call (OpenAI, Claude, etc.)
    
    Args:
        prompt (str): The prompt to send to AI
        
    Returns:
        str: AI response
    """
    # This is just a placeholder - implement your actual AI client here
    # Example for OpenAI:
    # response = openai.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    
    return "CASH_AND_CASH_EQUIVALENTS"  # Placeholder response

def main():
    """
    Example of how to use the balance sheet standardizer.
    """
    # Sample balance sheet data
    sample_data = {
        'Line Item': [
            'Cash and cash equivalents',
            'Trade receivables',
            'Inventory - finished goods',
            'Property, plant and equipment, net',
            'Goodwill and intangible assets',
            'Accounts payable',
            'Short-term borrowings',
            'Long-term debt',
            'Common stock',
            'Retained earnings'
        ],
        'Amount': [1000000, 500000, 750000, 2000000, 300000, 400000, 200000, 1500000, 100000, 800000]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize standardizer
    standardizer = BalanceSheetStandardizer('balance_sheet_prompt.json')
    
    # Process balance sheet by groups with reasoning
    print("Processing by Groups with Detailed Reasoning:")
    result_df, reasoning_data = standardizer.process_balance_sheet_by_groups(df, enhanced_example_ai_client)
    
    # Display results
    print("\nGroup-Based Processing Results:")
    print(result_df[['Line Item', 'Standardized_Category', 'Confidence']])
    
    # Save detailed reasoning analysis
    standardizer.save_reasoning_analysis(reasoning_data, 'balance_sheet_analysis.txt')
    standardizer.save_reasoning_json(reasoning_data, 'balance_sheet_reasoning.json')
    
    print(f"\nReasoning analysis saved to:")
    print(f"- Text format: balance_sheet_analysis.txt")
    print(f"- JSON format: balance_sheet_reasoning.json")
    
    # Show sample reasoning
    print(f"\nSample reasoning for debugging:")
    for item, details in list(reasoning_data.items())[:3]:
        print(f"\n{item}:")
        print(f"  Category: {details['category']}")
        print(f"  Confidence: {details['confidence']}")
        print(f"  Reasoning: {details['reasoning'][:100]}...")
    
    # Optional: Show group identification
    groups = standardizer.identify_balance_sheet_groups(df)
    print(f"\nIdentified Groups:")
    print(f"Assets: {len(groups['ASSETS'])} items")
    print(f"Liabilities & Equity: {len(groups['LIABILITIES_EQUITY'])} items")

if __name__ == "__main__":
    main()
