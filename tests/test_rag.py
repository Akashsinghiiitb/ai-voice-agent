import os
import unittest
from fastapi.testclient import TestClient

from crawler.crawler import WebCrawler
from pdf_parser.parser import PDFParser
from utils.helpers import clean_text, mask_pii, extract_html_structure
from vector_store.store import ChromaVectorStore
from ingestion.pipeline import IngestionPipeline
from backend.main import app

class TestRAGPipeline(unittest.TestCase):
    
    def test_cleaner_redundancies(self):
        raw_text = "Page 1 of 5 \n\n Clean  Spaces.  Home | About Us | Contact Us"
        cleaned = clean_text(raw_text)
        self.assertIn("Clean Spaces.", cleaned)
        self.assertNotIn("Page 1 of 5", cleaned)
        self.assertNotIn("Home | About Us", cleaned)

    def test_pii_redaction(self):
        text_with_pii = "Contact john.doe@gmail.com or call +1 555 123 4567 for info."
        masked, has_pii = mask_pii(text_with_pii)
        self.assertTrue(has_pii)
        self.assertIn("[MASKED_EMAIL]", masked)
        self.assertIn("[MASKED_PHONE]", masked)

    def test_html_parser_elements(self):
        html = "<html><body><h1>Policy Title</h1><p>Underwriting guidelines.</p></body></html>"
        struct = extract_html_structure(html)
        self.assertEqual(len(struct), 2)
        self.assertEqual(struct[0]["type"], "h1")
        self.assertEqual(struct[0]["content"], "Policy Title")

    def test_crawler_boundaries(self):
        crawler = WebCrawler("https://uiic.co.in", max_pages=2)
        self.assertEqual(crawler.domain, "uiic.co.in")
        self.assertTrue(crawler.is_valid_internal("https://uiic.co.in/health-insurance"))
        self.assertFalse(crawler.is_valid_internal("https://google.com"))

    def test_vector_store_mock_run(self):
        store = ChromaVectorStore()
        test_doc = [{
            "record_id": "test_chunk_0",
            "content": "Maternity coverage has a 9-month waiting period.",
            "title": "Maternity Guide",
            "category": "Policy PDF",
            "source": "maternity.pdf",
            "page": "3",
            "section": "Waiting Periods",
            "url": "local://maternity.pdf",
            "version": "1.0",
            "timestamp": "2026-07-17"
        }]
        store.add_documents(test_doc)
        res = store.query("waiting period for maternity", limit=1)
        self.assertTrue(len(res) > 0)
        self.assertEqual(res[0]["metadata"]["source"], "maternity.pdf")

    def test_api_ask_endpoint(self):
        client = TestClient(app)
        # Test non-empty query
        response = client.post("/ask", json={"question": "waiting period for cataract"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)

if __name__ == "__main__":
    unittest.main()
