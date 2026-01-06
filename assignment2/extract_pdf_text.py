import sys
from PyPDF2 import PdfReader

pdf_path = 'assignment2/Assignment2.pdf'
text_path = 'Assignment2.txt'

reader = PdfReader(pdf_path)
with open(text_path, 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        f.write(f"--- PAGE {i+1} ---\n")
        try:
            text = page.extract_text()
        except Exception as e:
            text = ''
        if text:
            f.write(text + '\n')
        else:
            f.write('\n')
print(f"Wrote extracted text to {text_path}")