from src.pdf_processor import PDFProcessor
from pathlib import Path

def main():
    # Initialize processor
    processor = PDFProcessor(model_path="models/output_ner_model")
    
    # Process single PDF
    pdf_path = "Axis 1-12-22 to 30-11-23.pdf"
    result = processor.process_single_pdf(pdf_path)
    
    # Print results
    if not result.error:
        print(f"\nProcessed: {result.filename}")
        print("\nExtracted Entities:")
        for entity in result.entities:
            print(f"  {entity.label}: {entity.text}")
        print("\nMetadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
    else:
        print(f"Error processing {result.filename}: {result.error}")
    
    # Export results
    processor.export_results("output_results.json")
    
    # Print summary
    summary = processor.get_summary()
    print("\nProcessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()