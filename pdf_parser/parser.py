import os
import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF (fitz) is not installed in the global environment.")

class PDFParser:
    """
    PDF parsing service that processes uploaded PDFs or downloads remote files
    and extracts text, layout structures, and page references using PyMuPDF.
    """
    def __init__(self):
        pass

    def download_pdf(self, url: str, dest_path: str) -> str:
        """
        Downloads a remote PDF file to a local destination path.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return dest_path

    def parse(self, file_path: str) -> list[dict]:
        """
        Parses a local PDF file page-by-page.
        Returns a list of dictionaries with page numbers, text content, and section headings.
        """
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is required to parse PDF documents.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        parsed_pages = []
        doc = fitz.open(file_path)
        
        print(f"Parsing PDF: {os.path.basename(file_path)} with {len(doc)} pages.")

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            
            # Simple heuristic for section heading: first short uppercase or bold line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            section = "General"
            for line in lines[:3]:
                if len(line) < 60 and (line.isupper() or "policy" in line.lower() or "schedule" in line.lower()):
                    section = line
                    break

            parsed_pages.append({
                "page": page_idx + 1,
                "content": text,
                "section": section,
                "source": os.path.basename(file_path)
            })

        doc.close()
        return parsed_pages
