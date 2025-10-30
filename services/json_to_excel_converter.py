#!/usr/bin/env python3
"""
Generic JSON Search Results to Excel Converter

This script can be used to convert any JSON search results file to a clean Excel format.
It automatically detects duplicates and applies professional styling.

Usage Examples:
    # Use default settings
    python json_to_excel_converter.py input.json
    
    # Specify custom output file
    python json_to_excel_converter.py input.json output.xlsx
    
    # Show help
    python json_to_excel_converter.py --help
"""

import json
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime

# Configuration - Modify these settings as needed
CONFIG = {
    'duplicate_field': 'link',  # Field to use for duplicate detection
    'sort_fields': ['date', 'title'],  # Fields to sort by (first field descending, rest ascending)
    'date_field': 'date',  # Field containing date information
    
    # Column mapping: original_field_name -> display_name
    'column_mapping': {
        'primary_alias': 'Company Name',
        'title': 'Title',
        'link': 'URL', 
        'snippet': 'Description',
        'source': 'Source',
        'date': 'Date',
        'search_engine': 'Search Engine',
        'used_query': 'Search Query Used'  # NEW FIELD: Shows which query found this article
    },
    
    # Excel styling
    'styling': {
        'header_color': '2F5F8F',  # Dark blue header
        'header_font_color': 'FFFFFF',  # White text
        'font_name': 'Calibri',
        'header_font_size': 11,
        'data_font_size': 10,
        'row_height': 40,
        'header_height': 25,
        
        # Column widths
        'column_widths': {
            'A': 20,  # Company Name
            'B': 60,  # Title
            'C': 80,  # URL
            'D': 80,  # Description
            'E': 25,  # Source
            'F': 15,  # Date
            'G': 15,  # Search Engine
            'H': 30,  # Search Query Used
        }
    }
}

def load_json_data(file_path):
    """Load and parse JSON data from file."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded JSON data from '{file_path}'")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"❌ Error: Failed to read '{file_path}': {e}")
        return None

def extract_search_results(data):
    """Extract search results from various JSON structures."""
    results = []
    primary_alias = "Unknown Company"
    
    # Try different common JSON structures
    if isinstance(data, list):
        # Simple list of results
        results = data
    elif 'results' in data:
        if 'search_results' in data['results']:
            # Structure: data.results.search_results
            results = data['results']['search_results']
        else:
            # Structure: data.results (direct list)
            results = data['results']
    elif 'search_results' in data:
        # Structure: data.search_results
        results = data['search_results']
    elif 'data' in data:
        # Structure: data.data
        results = data['data']
    else:
        # Assume the entire data is the results
        if isinstance(data, dict) and any(key in data for key in ['title', 'link', 'url']):
            results = [data]  # Single result
        else:
            results = []
    
    # Extract primary alias/company name from various locations
    if 'request_summary' in data and 'primary_alias' in data['request_summary']:
        primary_alias = data['request_summary']['primary_alias']
    elif 'company' in data:
        primary_alias = data['company']
    elif 'primary_alias' in data:
        primary_alias = data['primary_alias']
    elif results and 'company' in results[0]:
        primary_alias = results[0]['company']
    
    print(f"📊 Found {len(results)} total search results for '{primary_alias}'")
    return results, primary_alias

def deduplicate_results(results, duplicate_field='link'):
    """Remove duplicate entries based on specified field."""
    if not results:
        return results
    
    seen_values = set()
    unique_results = []
    duplicates_count = 0
    
    for result in results:
        value = result.get(duplicate_field, '')
        if value and value not in seen_values:
            seen_values.add(value)
            unique_results.append(result)
        else:
            duplicates_count += 1
    
    print(f"🧹 Removed {duplicates_count} duplicate entries")
    print(f"📝 {len(unique_results)} unique results remaining")
    
    return unique_results

def prepare_dataframe(results, primary_alias, column_mapping, sort_fields):
    """Convert search results to a pandas DataFrame."""
    if not results:
        return pd.DataFrame()
    
    data_for_df = []
    
    for result in results:
        row = {}
        for original_field, display_name in column_mapping.items():
            if original_field == 'primary_alias':
                row[display_name] = primary_alias
            else:
                # Handle different possible field names
                value = result.get(original_field, '')
                if not value and original_field == 'link':
                    value = result.get('url', '')  # Common alternative
                elif not value and original_field == 'snippet':
                    value = result.get('description', '')  # Common alternative
                row[display_name] = value
        data_for_df.append(row)
    
    df = pd.DataFrame(data_for_df)
    
    # Sort the data
    if sort_fields and all(CONFIG['column_mapping'].get(field) in df.columns for field in sort_fields):
        try:
            mapped_sort_fields = [CONFIG['column_mapping'][field] for field in sort_fields]
            
            # Create temporary column for date sorting if date field exists
            date_display_name = CONFIG['column_mapping'].get(CONFIG['date_field'])
            if date_display_name and date_display_name in df.columns:
                df['_temp_date'] = pd.to_datetime(df[date_display_name], errors='coerce')
                # Replace date field in sort with temp field
                if date_display_name in mapped_sort_fields:
                    temp_sort_fields = ['_temp_date' if f == date_display_name else f for f in mapped_sort_fields]
                    ascending = [False if f == '_temp_date' else True for f in temp_sort_fields]
                    df = df.sort_values(temp_sort_fields, ascending=ascending, na_position='last')
                    df = df.drop('_temp_date', axis=1)
                else:
                    df = df.sort_values(mapped_sort_fields, ascending=[False, True])
            else:
                df = df.sort_values(mapped_sort_fields)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not sort data: {e}")
            print("Data will be exported without sorting.")
    
    return df

def apply_professional_styling(file_path, styling_config):
    """Apply professional Excel styling."""
    try:
        wb = load_workbook(file_path)
        ws = wb.active
        
        # Define styles
        header_font = Font(
            name=styling_config['font_name'], 
            size=styling_config['header_font_size'], 
            bold=True, 
            color=styling_config['header_font_color']
        )
        header_fill = PatternFill(
            start_color=styling_config['header_color'], 
            end_color=styling_config['header_color'], 
            fill_type='solid'
        )
        header_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        
        data_font = Font(name=styling_config['font_name'], size=styling_config['data_font_size'])
        data_alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
        
        # Border style
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Style headers
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border
        
        # Style data rows
        for row in range(2, ws.max_row + 1):
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=row, column=col)
                cell.font = data_font
                cell.alignment = data_alignment
                cell.border = thin_border
        
        # Apply column widths
        for col_letter, width in styling_config['column_widths'].items():
            if ord(col_letter) - ord('A') < ws.max_column:
                ws.column_dimensions[col_letter].width = width
        
        # Set row heights
        ws.row_dimensions[1].height = styling_config['header_height']
        for row in range(2, ws.max_row + 1):
            ws.row_dimensions[row].height = styling_config['row_height']
        
        # Freeze the header row
        ws.freeze_panes = 'A2'
        
        # Save styled workbook
        wb.save(file_path)
        print(f"🎨 Applied professional styling to '{file_path}'")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not apply styling: {e}")

def export_to_excel(df, output_file, styling_config):
    """Export DataFrame to Excel with styling."""
    try:
        # Create Excel file
        df.to_excel(output_file, index=False, sheet_name='Search Results')
        print(f"📤 Exported {len(df)} results to '{output_file}'")
        
        # Apply styling
        apply_professional_styling(output_file, styling_config)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: Failed to export to Excel: {e}")
        return False

def print_summary(original_count, unique_count, duplicates_removed, primary_alias, output_file):
    """Print operation summary."""
    print("\n" + "="*70)
    print("📊 CONVERSION SUMMARY")
    print("="*70)
    print(f"🏢 Company: {primary_alias}")
    print(f"📥 Original entries: {original_count}")
    print(f"🗑️  Duplicates removed: {duplicates_removed}")
    print(f"📄 Unique entries exported: {unique_count}")
    print(f"💾 Output file: {output_file}")
    if original_count > 0:
        print(f"📈 Duplicate removal rate: {duplicates_removed/original_count*100:.1f}%")
    print("="*70)

def convert_json_to_excel(input_file, output_file=None, no_style=False):
    """Convert JSON to Excel - function version for API integration."""
    from pathlib import Path
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{input_path.stem}_cleaned_{timestamp}.xlsx"
    
    # Ensure output has .xlsx extension
    if not output_file.lower().endswith('.xlsx'):
        output_file += '.xlsx'
    
    print("🚀 JSON to Excel Converter")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print()
    
    # Load and process data
    data = load_json_data(input_file)
    if data is None:
        return None
    
    results, primary_alias = extract_search_results(data)
    original_count = len(results)
    
    if not results:
        print("❌ No results found to process.")
        return None
    
    # Skip deduplication since it's already handled by SERP endpoint
    print(f"📝 Processing {len(results)} unique results (deduplication already handled)")
    unique_results = results
    unique_count = len(unique_results)
    duplicates_removed = 0
    
    df = prepare_dataframe(unique_results, primary_alias, CONFIG['column_mapping'], CONFIG['sort_fields'])
    
    # Export to Excel
    styling_config = CONFIG['styling'] if not no_style else None
    success = export_to_excel(df, output_file, styling_config)
    
    if success:
        print_summary(original_count, unique_count, duplicates_removed, primary_alias, output_file)
        print(f"\n✅ Success! Data exported to: {output_file}")
        
        if not no_style:
            print("\n🎯 Features included:")
            print("  • Professional formatting with styled headers")
            print("  • Auto-adjusted column widths")  
            print("  • Frozen header row for easy navigation")
            print("  • Data sorted by date (newest first)")
        
        return output_file
    else:
        print("❌ Export failed. Check error messages above.")
        return None

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description='Convert JSON search results to professional Excel format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python json_to_excel_converter.py data.json
  python json_to_excel_converter.py data.json results.xlsx
  python json_to_excel_converter.py --help
        """
    )
    parser.add_argument(
        'input_file', 
        help='Input JSON file path'
    )
    parser.add_argument(
        'output_file', 
        nargs='?', 
        default=None,
        help='Output Excel file path (optional - auto-generated if not provided)'
    )
    parser.add_argument(
        '--no-style', 
        action='store_true',
        help='Skip Excel styling (faster processing)'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"{input_path.stem}_cleaned_{timestamp}.xlsx"
    
    # Ensure output has .xlsx extension
    if not args.output_file.lower().endswith('.xlsx'):
        args.output_file += '.xlsx'
    
    print("🚀 JSON to Excel Converter")
    print(f"📁 Input: {args.input_file}")
    print(f"📁 Output: {args.output_file}")
    print()
    
    # Load and process data
    data = load_json_data(args.input_file)
    if data is None:
        sys.exit(1)
    
    results, primary_alias = extract_search_results(data)
    original_count = len(results)
    
    if not results:
        print("❌ No results found to process.")
        sys.exit(1)
    
    unique_results = deduplicate_results(results, CONFIG['duplicate_field'])
    unique_count = len(unique_results)
    duplicates_removed = original_count - unique_count
    
    df = prepare_dataframe(unique_results, primary_alias, CONFIG['column_mapping'], CONFIG['sort_fields'])
    
    # Export to Excel
    styling_config = CONFIG['styling'] if not args.no_style else None
    success = export_to_excel(df, args.output_file, styling_config)
    
    if success:
        print_summary(original_count, unique_count, duplicates_removed, primary_alias, args.output_file)
        print(f"\n✅ Success! Data exported to: {args.output_file}")
        
        if not args.no_style:
            print("\n🎯 Features included:")
            print("  • Professional formatting with styled headers")
            print("  • Auto-adjusted column widths")  
            print("  • Frozen header row for easy navigation")
            print("  • Duplicate removal based on URL")
            print("  • Data sorted by date (newest first)")
        
        print(f"\n💡 Tip: You can customize the script by editing the CONFIG section")
    else:
        print("❌ Export failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
