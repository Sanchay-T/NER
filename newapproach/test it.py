from enhanced_extraction import BankStatementExtractor
import json

extractor = BankStatementExtractor("")
# Test with a single PDF
result = extractor.process_statement("bccb.pdf")
print(json.dumps(result, indent=2))
