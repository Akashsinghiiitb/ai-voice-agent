import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """
    Strips cookie warnings, footer items, navigation headers,
    and removes repeated menu bars or redundant blank lines.
    """
    # Remove menu-like strings (e.g. Home | About Us | Contact Us)
    text = re.sub(r'(\w+\s*\|\s*)+\w+', '', text)
    
    # Remove page number headers (e.g. Page 1 of 5)
    text = re.sub(r'(?i)\bpage\s+\d+\s+of\s+\d+\b', '', text)
    
    # Remove repeated copyright blocks
    text = re.sub(r'(?i)copyright\s+©\s+.*', '', text)
    
    # Remove cookie or privacy warning generic sentences
    text = re.sub(r'(?i)we use cookies to ensure that we give you the best experience.*', '', text)
    
    # Replace multiple newlines or double spacing with single spaces/newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize broken hyphen text (e.g. insur- \nance -> insurance)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    return text.strip()

def mask_pii(text: str) -> (str, bool):
    """
    Redacts emails, phone numbers, and SSNs.
    Returns masked text and a boolean flag indicating if PII was redacted.
    """
    has_pii = False
    
    # Redact Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    if re.search(email_pattern, text):
        text = re.sub(email_pattern, "[MASKED_EMAIL]", text)
        has_pii = True

    # Redact Phone Numbers
    phone_pattern = r'\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'
    if re.search(phone_pattern, text):
        text = re.sub(phone_pattern, "[MASKED_PHONE]", text)
        has_pii = True

    # Redact SSNs/National IDs
    id_pattern = r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
    if re.search(id_pattern, text):
        text = re.sub(id_pattern, "[MASKED_ID]", text)
        has_pii = True

    return text, has_pii

def extract_html_structure(html_content: str) -> list[dict]:
    """
    Parses HTML content, extracting headings, paragraphs, lists, and tables.
    Returns a structured list of text sections.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove navigation, headers, footers, style and script elements
    for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
        element.decompose()
        
    extracted_elements = []
    
    # Extract headings and paragraphs in linear order
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'table']):
        text = tag.get_text().strip()
        if not text:
            continue
            
        tag_type = tag.name
        
        if tag_type in ('ul', 'ol'):
            # Convert list items to bullet markdown lines
            items = [f"- {li.get_text().strip()}" for li in tag.find_all('li') if li.get_text().strip()]
            text = "\n".join(items)
            
        elif tag_type == 'table':
            # Convert simple tables to markdown format
            rows = []
            for tr in tag.find_all('tr'):
                cols = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if any(cols):
                    rows.append("| " + " | ".join(cols) + " |")
            if rows:
                text = "\n".join(rows)
            else:
                continue

        extracted_elements.append({
            "type": tag_type,
            "content": text
        })
        
    return extracted_elements
