import sys
import json
import os

try:
    # pyrefly: ignore [missing-import]
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Docling is not installed. Please install it using 'pip install docling'.")
    sys.exit(1)

def main():
    # 1. We require two arguments: the PDF to read, and the JSON file to write
    if len(sys.argv) != 3:
        print("Usage: python parse.py <input_pdf> <output_json>")
        sys.exit(1)
        
    input_pdf = sys.argv[1]
    output_json = sys.argv[2]
    
    print(f"Parsing PDF: {input_pdf}")
    
    # 2. Instantiate the core engine. This object holds the AI models for layout parsing.
    converter = DocumentConverter()
    
    # 3. The actual conversion step. Docling scans the PDF, identifies tables/headings/paragraphs,
    # and builds a logical Document tree in memory.
    result = converter.convert(input_pdf)
    
    # 4. Serialize the Document tree into a standard Python dictionary so we can save it.
    doc_dict = result.document.export_to_dict()
    
    # 5. Ensure the output directory (data/parsed/) exists before trying to save the file.
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # 6. Save it as a standard JSON file. This caches the extraction so our chunker can
    # run instantly on the JSON rather than waiting 5 minutes for the PDF parser every time.
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(doc_dict, f, indent=2)
        
    print(f"Successfully saved parsed JSON to: {output_json}")

if __name__ == "__main__":
    main()
