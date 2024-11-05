from openai import OpenAI
import fitz
import re
import unicodedata
from typing import Dict, Any, List, Tuple
import json
import os


class BankStatementExtractor:
    def __init__(self, openai_key: str):
        print("\n=== Initializing BankStatementExtractor ===")
        self.client = OpenAI(api_key=openai_key)
        print("✓ OpenAI client initialized")

    def extract_text(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction with better cleaning"""
        print("\n=== Starting PDF Text Extraction ===")
        print(f"Processing PDF: {pdf_path}")

        def clean_text(text: str) -> str:
            print("\nCleaning text...")
            original_length = len(text)

            # Normalize unicode characters
            text = unicodedata.normalize("NFKD", text)

            # Remove special characters but preserve important formatting
            text = re.sub(r"[^\x20-\x7E\n:]", " ", text)

            # Normalize whitespace but preserve line breaks
            text = re.sub(r" +", " ", text)
            text = re.sub(r"\n+", "\n", text)

            cleaned_length = len(text.strip())
            print(f"Text length: {original_length} → {cleaned_length} characters")
            return text.strip()

        try:
            with fitz.open(pdf_path) as doc:
                print(f"PDF opened successfully: {doc.page_count} pages")
                full_text = ""

                for page_num in range(doc.page_count):
                    print(f"\nProcessing page {page_num + 1}/{doc.page_count}")
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    cleaned_text = clean_text(text)

                    # Check for transaction table markers
                    table_markers = [
                        r"(?i)^.*?(Date|Transaction|Particulars|Description)",
                        r"^\d{2}[-/]\d{2}[-/]\d{2,4}",
                        r"^Balance brought forward",
                    ]

                    # Split text at first table marker
                    for marker in table_markers:
                        match = re.search(marker, cleaned_text, re.MULTILINE)
                        if match:
                            print(
                                f"Found table marker: '{marker}' at position {match.start()}"
                            )
                            full_text += cleaned_text[: match.start()]
                            print("\nExtracted header text:")
                            print("-------------------")
                            print(full_text.strip())
                            print("-------------------")
                            return full_text.strip()

                    full_text += cleaned_text + "\n"

                print("\nNo table markers found, using full text")
                print("-------------------")
                print(full_text.strip())
                print("-------------------")
                return full_text.strip()

        except Exception as e:
            print(f"❌ Error during PDF extraction: {str(e)}")
            raise

    def get_entities_from_llm(self, text: str) -> Dict[str, str]:
        """Use GPT to identify account numbers and names without positions"""
        print("\n=== Starting LLM Entity Extraction ===")

        system_prompt = """You are a precise bank statement information extractor. 
        Extract ONLY the account number and account holder name from bank statements.
        Return ONLY a JSON object with exactly these fields: account_number and name."""

        user_prompt = f"""Extract ONLY the account number and account holder name from this text.
        Return ONLY a JSON object like this: {{"account_number": "number", "name": "holder name"}}
        
        Text:
        {text}"""

        print("\nSending request to OpenAI API...")
        print(f"System prompt length: {len(system_prompt)}")
        print(f"User prompt length: {len(user_prompt)}")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=150,
            )

            print("\nReceived response from OpenAI")

            if not response.choices:
                print("❌ No choices in response")
                raise ValueError("No choices in response")

            content = response.choices[0].message.content.strip()
            print("\nRaw response content:")
            print("-------------------")
            print(content)
            print("-------------------")

            try:
                result = json.loads(content)
                print("\nSuccessfully parsed JSON response")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print("Attempting to clean malformed JSON...")
                cleaned_content = (
                    content.replace("\n", "").replace("```json", "").replace("```", "")
                )
                result = json.loads(cleaned_content)
                print("✓ Successfully parsed cleaned JSON")

            # Validate and clean the result
            required_fields = ["account_number", "name"]
            if not all(field in result for field in required_fields):
                print(f"❌ Missing required fields. Got: {list(result.keys())}")
                raise ValueError(f"Missing required fields. Got: {list(result.keys())}")

            result = {
                "account_number": result["account_number"].strip(),
                "name": result["name"].strip(),
            }

            print("\nExtracted entities:")
            print(f"Account Number: '{result['account_number']}'")
            print(f"Name: '{result['name']}'")

            return result

        except Exception as e:
            print(f"❌ Error during LLM extraction: {str(e)}")
            raise

    def process_statement(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single statement"""
        print("\n=== Starting Statement Processing ===")

        try:
            # Extract text
            text = self.extract_text(pdf_path)

            # Get entities from LLM
            entities = self.get_entities_from_llm(text)

            # Find exact positions
            print("\nFinding exact positions of entities...")
            annotations = []

            # Find account number positions
            if "account_number" in entities:
                acc_no = entities["account_number"]
                start = text.find(acc_no)
                if start != -1:
                    annotations.append(
                        {
                            "text": acc_no,
                            "start": start,
                            "end": start + len(acc_no),
                            "label": "ACC_NO",
                        }
                    )
                    print(f"Found account number '{acc_no}' at position {start}")
                else:
                    print(
                        f"❌ Could not find exact position for account number: {acc_no}"
                    )

            # Find name positions
            if "name" in entities:
                name = entities["name"]
                start = text.find(name)
                if start != -1:
                    annotations.append(
                        {
                            "text": name,
                            "start": start,
                            "end": start + len(name),
                            "label": "PER",
                        }
                    )
                    print(f"Found name '{name}' at position {start}")
                else:
                    print(f"❌ Could not find exact position for name: {name}")

            print("\nFinal annotations:")
            print(json.dumps(annotations, indent=2))

            return {"text": text, "entities": annotations}

        except Exception as e:
            print(f"❌ Error processing statement: {str(e)}")
            return {"error": str(e)}


def main():
    print("\n=== Starting Bank Statement Processing ===")

    # Get API key from environment variable
    api_key = ""
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    try:
        extractor = BankStatementExtractor(api_key)
        result = extractor.process_statement("bccb.pdf")

        if "error" not in result:
            print("\n=== Final Results ===")
            print("\nExtracted Text:")
            print("-------------------")
            print(result["text"])
            print("-------------------")

            print("\nFound Entities:")
            for entity in result["entities"]:
                print(
                    f"{entity['label']}: '{entity['text']}' at positions {entity['start']}:{entity['end']}"
                )
        else:
            print(f"\n❌ Error: {result['error']}")

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
