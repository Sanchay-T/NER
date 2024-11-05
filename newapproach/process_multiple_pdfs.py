import os
from enhanced_extraction import BankStatementExtractor
import json
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict
import pandas as pd


class BatchProcessor:
    def __init__(self, openai_key: str):
        print("\n=== Initializing Batch Processor ===")
        self.extractor = BankStatementExtractor(openai_key)
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0

    def is_password_protected(self, pdf_path: str) -> bool:
        """Check if PDF is password protected"""
        try:
            doc = fitz.open(pdf_path)
            is_protected = doc.is_encrypted
            doc.close()
            return is_protected
        except Exception:
            return True  # Assume protected if can't open

    def process_directory(self, base_dir: str) -> List[Dict]:
        """Process all PDFs in directory and subdirectories"""
        print(f"\n=== Starting Batch Processing ===")
        print(f"Base directory: {base_dir}")

        all_results = []
        failed_files = []
        skipped_files = []

        # Walk through directory
        for root, _, files in os.walk(base_dir):
            bank_name = os.path.basename(root)
            pdf_files = [f for f in files if f.lower().endswith(".pdf")]

            if pdf_files:
                print(f"\nProcessing bank: {bank_name}")
                print(f"Found {len(pdf_files)} PDF files")

                for pdf_file in pdf_files:
                    pdf_path = os.path.join(root, pdf_file)
                    print(f"\nProcessing: {pdf_file}")

                    # Check if password protected
                    if self.is_password_protected(pdf_path):
                        print(f"❌ Skipping password protected file: {pdf_file}")
                        skipped_files.append(
                            {
                                "file": pdf_file,
                                "bank": bank_name,
                                "reason": "Password protected",
                            }
                        )
                        self.skipped_count += 1
                        continue

                    try:
                        # Process the PDF
                        result = self.extractor.process_statement(pdf_path)

                        if "error" not in result:
                            # Add metadata
                            result["metadata"] = {
                                "filename": pdf_file,
                                "bank": bank_name,
                                "path": pdf_path,
                            }
                            all_results.append(result)
                            self.processed_count += 1
                            print(f"✓ Successfully processed: {pdf_file}")
                        else:
                            print(f"❌ Failed to process: {pdf_file}")
                            failed_files.append(
                                {
                                    "file": pdf_file,
                                    "bank": bank_name,
                                    "error": result["error"],
                                }
                            )
                            self.failed_count += 1

                    except Exception as e:
                        print(f"❌ Error processing {pdf_file}: {str(e)}")
                        failed_files.append(
                            {"file": pdf_file, "bank": bank_name, "error": str(e)}
                        )
                        self.failed_count += 1

        # Save processing reports
        self.save_reports(all_results, failed_files, skipped_files)

        return all_results

    def save_reports(
        self, results: List[Dict], failed: List[Dict], skipped: List[Dict]
    ):
        """Save processing results and reports"""
        output_dir = "processing_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save successful results
        if results:
            with open(
                f"{output_dir}/successful_extractions.json", "w", encoding="utf-8"
            ) as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Create spaCy training format
            spacy_format = []
            for result in results:
                spacy_format.append(
                    {
                        "text": result["text"],
                        "entities": [
                            [ent["start"], ent["end"], ent["label"]]
                            for ent in result["entities"]
                        ],
                    }
                )

            with open(
                f"{output_dir}/spacy_training_data.json", "w", encoding="utf-8"
            ) as f:
                json.dump(spacy_format, f, ensure_ascii=False, indent=2)

        # Save failed files report
        if failed:
            pd.DataFrame(failed).to_csv(f"{output_dir}/failed_files.csv", index=False)

        # Save skipped files report
        if skipped:
            pd.DataFrame(skipped).to_csv(f"{output_dir}/skipped_files.csv", index=False)

        # Save summary report
        summary = {
            "total_processed": self.processed_count,
            "successful": self.processed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
        }

        with open(f"{output_dir}/processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Processing Summary ===")
        print(f"Total processed: {self.processed_count}")
        print(f"Successful: {self.processed_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Skipped: {self.skipped_count}")
        print(f"\nResults saved in: {output_dir}/")


def main():
    # Get API key from environment variable
    api_key = ""
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Initialize processor
    processor = BatchProcessor(api_key)

    # Process PDFs from parent directory
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "SHORT_PAGES_BANK_STATEMENTS -2"
    )

    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return

    # Process all PDFs
    processor.process_directory(base_dir)


if __name__ == "__main__":
    main()
