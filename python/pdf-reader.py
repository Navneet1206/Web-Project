import pdfplumber

# Replace 'your_file.pdf' with the path to your PDF file
file_path = 'example/Solidity Basics_ Keywords Explained.pdf'

# Open the PDF file
with pdfplumber.open(file_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

# Display the extracted text
print(text)
