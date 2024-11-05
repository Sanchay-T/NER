from src.pdf_processor import PDFProcessor
from pathlib import Path
import json
from datetime import datetime
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("pdf_processing.log"), logging.StreamHandler()],
    )


def process_statement_folders(
    base_folder="SHORT_PAGES_BANK_STATEMENTS -2",
    output_file="processing_results.json",
    summary_file="processing_summary.json",
):
    # Initialize processor
    processor = PDFProcessor(model_path="models/output_ner_model")

    # Initialize results storage
    all_results = []

    # Get base folder path
    base_path = Path(base_folder)

    # Process each bank folder
    for bank_folder in base_path.iterdir():
        if bank_folder.is_dir():
            logging.info(f"Processing bank folder: {bank_folder.name}")

            # Process each PDF in the bank folder
            for pdf_file in bank_folder.glob("*.pdf"):
                try:
                    # Attempt to process the PDF
                    result = processor.process_single_pdf(str(pdf_file))

                    if result.error:
                        if "password protected" in str(result.error).lower():
                            logging.warning(
                                f"Skipping password-protected PDF: {pdf_file.name}"
                            )
                            continue
                        else:
                            logging.error(
                                f"Error processing {pdf_file.name}: {result.error}"
                            )
                            continue

                    # Create detailed result dictionary
                    processed_result = {
                        "timestamp": datetime.now().isoformat(),
                        "bank_name": bank_folder.name,
                        "pdf_name": pdf_file.name,
                        "pdf_path": str(pdf_file),
                        "extracted_text": result.raw_text,  # Add the raw extracted text
                        "entities": {
                            entity.label: {
                                "text": entity.text,
                                "confidence": entity.confidence
                                if hasattr(entity, "confidence")
                                else None,
                                "position": entity.position
                                if hasattr(entity, "position")
                                else None,
                            }
                            for entity in result.entities
                        },
                        "metadata": result.metadata,
                        "processing_status": "success",
                    }

                    all_results.append(processed_result)
                    logging.info(f"Successfully processed: {pdf_file.name}")

                except Exception as e:
                    logging.error(
                        f"Unexpected error processing {pdf_file.name}: {str(e)}"
                    )
                    all_results.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "bank_name": bank_folder.name,
                            "pdf_name": pdf_file.name,
                            "pdf_path": str(pdf_file),
                            "error": str(e),
                            "processing_status": "failed",
                        }
                    )

    # Export results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate and print summary
    summary = generate_summary(all_results)

    # Save detailed summary to a separate file
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print concise summary to console
    print("\nProcessing Summary:")
    print(f"  Total PDFs Processed: {summary['Total PDFs Processed']}")
    print(f"  Successfully Processed: {summary['Successfully Processed']}")
    print(f"  Failed: {summary['Failed']}")
    print(f"  Success Rate: {summary['Success Rate']}")
    print("\nEntity Statistics:")
    print("  PERSON (PER):")
    print(f"    Found: {summary['Entity Statistics']['PERSON (PER)']['Found']}")
    print(f"    Missing: {summary['Entity Statistics']['PERSON (PER)']['Missing']}")
    print(f"    Success Rate: {summary['Entity Statistics']['PERSON (PER)']['Success Rate']}")
    print("  ACCOUNT NUMBER (ACC NO):")
    print(f"    Found: {summary['Entity Statistics']['ACCOUNT NUMBER (ACC NO)']['Found']}")
    print(f"    Missing: {summary['Entity Statistics']['ACCOUNT NUMBER (ACC NO)']['Missing']}")
    print(f"    Success Rate: {summary['Entity Statistics']['ACCOUNT NUMBER (ACC NO)']['Success Rate']}")
    
    print("\nExtracted Entities:")
    if summary["Extracted Entities"]["PERSON"]:
        print("\n  PERSON entities found:")
        for entry in summary["Extracted Entities"]["PERSON"]:
            print(f"    {entry['file']}: {entry['text']}")
    
    if summary["Extracted Entities"]["ACCOUNT NUMBER"]:
        print("\n  ACCOUNT NUMBER entities found:")
        for entry in summary["Extracted Entities"]["ACCOUNT NUMBER"]:
            print(f"    {entry['file']}: {entry['text']}")
    
    print("\nDetailed summary saved to:", summary_file)

    return all_results


def generate_summary(results):
    """Generate a detailed summary of the processing results"""
    total_pdfs = len(results)
    successful = sum(1 for r in results if r.get("processing_status") == "success")
    failed = total_pdfs - successful
    banks_processed = len(set(r.get("bank_name") for r in results))

    # Initialize entity tracking
    entity_stats = {
        "PER": {"found": 0, "missing": 0, "files_missing": []},
        "ACC NO": {"found": 0, "missing": 0, "files_missing": []},
        "high_confidence_matches": [],
        "low_confidence_matches": [],
    }

    # Analyze successful results for entity statistics
    for result in results:
        if result.get("processing_status") == "success":
            entities = result.get("entities", {})
            pdf_name = result.get("pdf_name")

            # Check PER entities
            if "PER" in entities:
                entity_stats["PER"]["found"] += 1
                confidence = entities["PER"].get("confidence", 0)
                if confidence and confidence > 0.8:
                    entity_stats["high_confidence_matches"].append({
                        "file": pdf_name,
                        "entity": "PER",
                        "text": entities["PER"]["text"],
                        "confidence": confidence
                    })
                else:
                    entity_stats["low_confidence_matches"].append({
                        "file": pdf_name,
                        "entity": "PER",
                        "text": entities["PER"]["text"],
                        "confidence": confidence
                    })
            else:
                entity_stats["PER"]["missing"] += 1
                entity_stats["PER"]["files_missing"].append(pdf_name)

            # Check ACC NO entities
            if "ACC NO" in entities:
                entity_stats["ACC NO"]["found"] += 1
                confidence = entities["ACC NO"].get("confidence", 0)
                if confidence and confidence > 0.8:
                    entity_stats["high_confidence_matches"].append({
                        "file": pdf_name,
                        "entity": "ACC NO",
                        "text": entities["ACC NO"]["text"],
                        "confidence": confidence
                    })
                else:
                    entity_stats["low_confidence_matches"].append({
                        "file": pdf_name,
                        "entity": "ACC NO",
                        "text": entities["ACC NO"]["text"],
                        "confidence": confidence
                    })
            else:
                entity_stats["ACC NO"]["missing"] += 1
                entity_stats["ACC NO"]["files_missing"].append(pdf_name)

    summary = {
        "Total PDFs Processed": total_pdfs,
        "Successfully Processed": successful,
        "Failed": failed,
        "Banks Processed": banks_processed,
        "Success Rate": f"{(successful/total_pdfs)*100:.2f}%" if total_pdfs > 0 else "0%",
        "Entity Statistics": {
            "PERSON (PER)": {
                "Found": entity_stats["PER"]["found"],
                "Missing": entity_stats["PER"]["missing"],
                "Success Rate": f"{(entity_stats['PER']['found']/successful)*100:.2f}%" if successful > 0 else "0%"
            },
            "ACCOUNT NUMBER (ACC NO)": {
                "Found": entity_stats["ACC NO"]["found"],
                "Missing": entity_stats["ACC NO"]["missing"],
                "Success Rate": f"{(entity_stats['ACC NO']['found']/successful)*100:.2f}%" if successful > 0 else "0%"
            }
        }
    }

    # Add detailed missing files information
    if entity_stats["PER"]["files_missing"]:
        summary["Missing PERSON in Files"] = entity_stats["PER"]["files_missing"]
    if entity_stats["ACC NO"]["files_missing"]:
        summary["Missing ACCOUNT NUMBER in Files"] = entity_stats["ACC NO"]["files_missing"]

    # Add confidence information
    if entity_stats["high_confidence_matches"]:
        summary["High Confidence Matches (>80%)"] = entity_stats["high_confidence_matches"]
    if entity_stats["low_confidence_matches"]:
        summary["Low Confidence Matches (≤80%)"] = entity_stats["low_confidence_matches"]

    # Add extracted entities with text
    summary["Extracted Entities"] = {
        "PERSON": [],
        "ACCOUNT NUMBER": []
    }
    
    for result in results:
        if result.get("processing_status") == "success":
            entities = result.get("entities", {})
            if "PER" in entities:
                summary["Extracted Entities"]["PERSON"].append({
                    "file": result["pdf_name"],
                    "text": entities["PER"]["text"]
                })
            if "ACC NO" in entities:
                summary["Extracted Entities"]["ACCOUNT NUMBER"].append({
                    "file": result["pdf_name"],
                    "text": entities["ACC NO"]["text"]
                })

    return summary


if __name__ == "__main__":
    setup_logging()
    process_statement_folders()
