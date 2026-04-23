import os
try:
    import fitz  # PyMuPDF
    doc = fitz.open('spliteed.pdf')
    text = ''
    for page in doc:
        text += page.get_text()
    with open('spliteed.txt', 'w') as f:
        f.write(text)
    print("PyMuPDF success")
except ImportError:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader('spliteed.pdf')
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        with open('spliteed.txt', 'w') as f:
            f.write(text)
        print("PyPDF2 success")
    except ImportError:
        print("No suitable PDF library found. Please install PyMuPDF or PyPDF2.")
