import fitz
import spacy
import json
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, List
from .data_models import ExtractedEntity, ProcessedDocument
import re

class PDFProcessor:
    def __init__(self, model_path: str = "output_ner_model"):
        """Initialize the PDF processor with the trained spaCy model"""
        try:
            # Disable GPU and use CPU only
            spacy.require_cpu()
            self.nlp = spacy.load(model_path)
            self.processed_documents = []
        except Exception as e:
            raise Exception(f"Failed to load spaCy model: {str(e)}")

    def process_single_pdf(self, pdf_path: Union[str, Path]) -> ProcessedDocument:
        """Process a single PDF file and return structured results"""
        try:
            pdf_path = Path(pdf_path)
            timestamp = datetime.now().isoformat()
            
            # Extract text from PDF
            extracted_data = self._extract_text_from_pdf(pdf_path)
            
            if "error" in extracted_data:
                return ProcessedDocument(
                    filename=pdf_path.name,
                    timestamp=timestamp,
                    raw_text="",
                    preprocessed_text="",
                    entities=[],
                    metadata={},
                    error=extracted_data["error"]
                )
            
            # Process with spaCy
            entities = self._process_with_spacy(extracted_data["preprocessed_text"])
            
            # Create processed document
            processed_doc = ProcessedDocument(
                filename=pdf_path.name,
                timestamp=timestamp,
                raw_text=extracted_data["raw_text"],
                preprocessed_text=extracted_data["preprocessed_text"],
                entities=entities,
                metadata={
                    "account_number": extracted_data["account_number"],
                    "person_name": extracted_data["person_name"],
                    "file_path": str(pdf_path.absolute())
                },
                header_text=extracted_data.get("header_text"),
                header_file_path=extracted_data.get("header_file_path")
            )
            
            # Store processed document
            self.processed_documents.append(processed_doc)
            return processed_doc
            
        except Exception as e:
            return ProcessedDocument(
                filename=pdf_path.name,
                timestamp=datetime.now().isoformat(),
                raw_text="",
                preprocessed_text="",
                entities=[],
                metadata={},
                error=str(e)
            )

    def _process_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Process text with spaCy model and return structured entities"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char
            )
            entities.append(entity)
            
        return entities

    def process_multiple_pdfs(self, pdf_dir: Union[str, Path]) -> List[ProcessedDocument]:
        """Process all PDFs in a directory"""
        pdf_dir = Path(pdf_dir)
        results = []
        
        for pdf_file in pdf_dir.glob("**/*.pdf"):
            result = self.process_single_pdf(pdf_file)
            results.append(result)
            
        return results

    def export_results(self, output_path: Union[str, Path], format: str = "json"):
        """Export processing results to specified format"""
        output_path = Path(output_path)
        
        if format.lower() == "json":
            self._export_to_json(output_path)
        elif format.lower() == "excel":
            self._export_to_excel(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_to_json(self, output_path: Path):
        """Export results to JSON format"""
        results = []
        for doc in self.processed_documents:
            doc_dict = {
                "filename": doc.filename,
                "timestamp": doc.timestamp,
                "metadata": doc.metadata,
                "entities": [
                    {
                        "text": ent.text,
                        "label": ent.label,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    }
                    for ent in doc.entities
                ],
                "error": doc.error
            }
            results.append(doc_dict)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def get_summary(self) -> Dict:
        """Get summary of processed documents"""
        return {
            "total_documents": len(self.processed_documents),
            "successful": len([doc for doc in self.processed_documents if not doc.error]),
            "failed": len([doc for doc in self.processed_documents if doc.error]),
            "total_entities": sum(len(doc.entities) for doc in self.processed_documents)
        }

    def _extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        try:
            print(f"Starting to process PDF: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            first_page_text = first_page.get_text()
            doc.close()
            
            table_indicators = [
                r"Date\s+Dr Amount\s+Cr Amount\s+Total Amount",  
                r"Date\s+Particulars\s+Instruments",  
                r"\d{2}[-/]\d{2}[-/]\d{4}\s+\d+\.\d{2}\s+",  
                r"Opening Balance.*?Closing Balance",
                r"Transaction Details:"
            ]

            # Find where table starts
            table_start_pos = len(first_page_text)
            matched_pattern = None
            
            for pattern in table_indicators:
                match = re.search(pattern, first_page_text, re.IGNORECASE | re.MULTILINE)
                if match and match.start() < table_start_pos:
                    table_start_pos = match.start()
                    matched_pattern = pattern
                    print(f"Found table start with pattern: {pattern}")
                    print(f"At position: {table_start_pos}")
                    print(f"Text around match point:\n{first_page_text[max(0, table_start_pos-50):table_start_pos+50]}")

            # Get header content
            header_content = first_page_text[:table_start_pos].strip()
            
            # Create output directory and save
            output_dir = Path("extracted_text")
            output_dir.mkdir(exist_ok=True)
            text_file_path = output_dir / f"{pdf_path.stem}_header.txt"
            
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(header_content)
            
            print(f"\nExtracted header length: {len(header_content)}")
            print(f"First 100 chars: {header_content[:100]}")
            print(f"Last 100 chars: {header_content[-100:]}")
            
            return {
                "raw_text": header_content,
                "preprocessed_text": header_content,
                "account_number": "",
                "person_name": "",
                "extracted_file_path": str(text_file_path)
            }
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return {"error": f"Failed to extract text from PDF: {str(e)}"}