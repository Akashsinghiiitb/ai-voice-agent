import time
import uuid
import datetime
from crawler.crawler import WebCrawler
from pdf_parser.parser import PDFParser
from utils.helpers import clean_text, mask_pii, extract_html_structure
from vector_store.store import ChromaVectorStore


class IngestionPipeline:
    """
    Orchestrates the entire ingestion cycle: crawls web targets, downloads PDFs,
    scrubs layout and PII anomalies, chunks text, and embeds into ChromaDB.
    """
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.pdf_parser = PDFParser()

    def split_content(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """
        Splits content recursively by paragraphs, newlines, spaces, or characters
        to match RecursiveCharacterTextSplitter behavior without requiring LangChain.
        """
        separators = ["\n\n", "\n", " ", ""]
        
        def _split(t: str, seps: list[str]) -> list[str]:
            if len(t) <= chunk_size:
                return [t]
            if not seps:
                return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size - overlap)]
                
            sep = seps[0]
            parts = t.split(sep)
            chunks = []
            current = []
            current_len = 0
            
            for part in parts:
                part_len = len(part)
                if part_len > chunk_size:
                    if current:
                        chunks.append(sep.join(current))
                        current = []
                        current_len = 0
                    chunks.extend(_split(part, seps[1:]))
                elif current_len + part_len + (len(sep) if current else 0) <= chunk_size:
                    current.append(part)
                    current_len += part_len + (len(sep) if len(current) > 1 else 0)
                else:
                    if current:
                        chunks.append(sep.join(current))
                    current = [part]
                    current_len = part_len
                    
            if current:
                chunks.append(sep.join(current))
                
            # Reconstruct chunks with overlap
            final_chunks = []
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    final_chunks.append(chunk)
                else:
                    prev = chunks[idx - 1]
                    overlap_part = prev[-overlap:] if len(prev) > overlap else prev
                    final_chunks.append(overlap_part + " " + chunk)
            return final_chunks

        return _split(text, separators)

    def run(self, start_url: str = None, local_pdf_paths: list[str] = None) -> dict:
        """
        Runs the full ingestion flow. Returns statistics on parsed components.
        """
        raw_documents = []
        crawled_urls = 0
        pdf_pages_parsed = 0
        
        # 1. Handle Crawling Domain
        if start_url:
            crawler = WebCrawler(start_url, max_pages=8)
            crawl_results = crawler.crawl()
            
            # Parse HTML content
            for url, html_content in crawl_results["pages"].items():
                crawled_urls += 1
                elements = extract_html_structure(html_content)
                
                # Combine elements to document structures
                page_text = "\n\n".join([el["content"] for el in elements])
                cleaned = clean_text(page_text)
                masked, has_pii = mask_pii(cleaned)
                
                raw_documents.append({
                    "title": f"Web: {url.replace('https://', '')[:40]}",
                    "content": masked,
                    "category": "Website FAQ",
                    "source": "Website",
                    "page": "1",
                    "section": "Web Page Content",
                    "url": url,
                    "has_pii": has_pii
                })
                
            # Download and parse remote discovered PDFs
            for idx, pdf_url in enumerate(crawl_results["pdfs"][:3]):
                try:
                    local_tmp = f"tmp_crawled_doc_{idx}.pdf"
                    print(f"Downloading remote PDF: {pdf_url}")
                    self.pdf_parser.download_pdf(pdf_url, local_tmp)
                    
                    pages = self.pdf_parser.parse(local_tmp)
                    for p in pages:
                        pdf_pages_parsed += 1
                        cleaned = clean_text(p["content"])
                        masked, has_pii = mask_pii(cleaned)
                        
                        raw_documents.append({
                            "title": p["title"] if "title" in p else p["source"],
                            "content": masked,
                            "category": "Brochure",
                            "source": p["source"],
                            "page": str(p["page"]),
                            "section": p["section"],
                            "url": pdf_url,
                            "has_pii": has_pii
                        })
                except Exception as e:
                    print(f"Failed parsing downloaded PDF {pdf_url}: {e}")
                    
        # 2. Handle Uploaded PDFs
        if local_pdf_paths:
            for path in local_pdf_paths:
                try:
                    pages = self.pdf_parser.parse(path)
                    for p in pages:
                        pdf_pages_parsed += 1
                        cleaned = clean_text(p["content"])
                        masked, has_pii = mask_pii(cleaned)
                        
                        raw_documents.append({
                            "title": f"PDF: {os.path.basename(path)}",
                            "content": masked,
                            "category": "Policy PDF",
                            "source": p["source"],
                            "page": str(p["page"]),
                            "section": p["section"],
                            "url": f"local://{os.path.basename(path)}",
                            "has_pii": has_pii
                        })
                except Exception as e:
                    print(f"Failed parsing local PDF {path}: {e}")
                    
        # 3. Chunking & Database ingestion
        chunks_to_insert = []
        for doc in raw_documents:
            text_chunks = self.split_content(doc["content"])
            for idx, chunk in enumerate(text_chunks):
                record_id = f"kb_{str(uuid.uuid4())[:8]}_chunk_{idx}"
                chunks_to_insert.append({
                    "record_id": record_id,
                    "title": doc["title"],
                    "content": chunk,
                    "category": doc["category"],
                    "source": doc["source"],
                    "page": doc["page"],
                    "section": doc["section"],
                    "url": doc["url"],
                    "version": "1.0",
                    "timestamp": str(datetime.datetime.utcnow())
                })
                
        if chunks_to_insert:
            self.vector_store.add_documents(chunks_to_insert)
            
        return {
            "status": "success",
            "crawled_urls": crawled_urls,
            "pdf_pages_parsed": pdf_pages_parsed,
            "total_chunks_ingested": len(chunks_to_insert)
        }
