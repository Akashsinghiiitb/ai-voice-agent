import re
import time
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

class WebCrawler:
    """
    Web Crawler module that crawls within a target domain (e.g. uiic.co.in),
    discovers internal links, filters out noise pages (cookies, privacy, news),
    and identifies PDF links.
    """
    def __init__(self, base_url: str, max_pages: int = 15):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited_urls = set()
        self.pdf_urls = set()
        self.crawled_data = {} # Maps URL -> raw HTML/text

        # Keywords to filter out noise pages
        self.exclude_patterns = [
            r'/privacy', r'/cookie', r'/career', r'/investor', 
            r'/news', r'/tender', r'/contact', r'/feedback',
            r'disclaimer', r'terms-of-use'
        ]
        
        # Keywords to prioritize
        self.priority_keywords = [
            'health', 'insurance', 'policy', 'coverage', 'benefit', 
            'eligibility', 'faq', 'waiting', 'claim', 'cashless', 
            'download', 'brochure', 'terms'
        ]

    def is_valid_internal(self, url: str) -> bool:
        """
        Validates if the URL is internal to the target domain, is HTTP/S,
        and does not match exclusion patterns.
        """
        parsed = urlparse(url)
        # Verify domain matches
        if parsed.netloc and parsed.netloc != self.domain:
            return False
            
        # Verify scheme is web-based
        if parsed.scheme not in ('http', 'https'):
            return False
            
        # Filter exclusions
        for pattern in self.exclude_patterns:
            if re.search(pattern, parsed.path, re.IGNORECASE):
                return False
                
        return True

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """
        Extracts all internal links and isolates PDF downloads.
        """
        found_links = []
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = urljoin(current_url, href)
            
            # Remove url fragments
            full_url = full_url.split('#')[0]
            
            if full_url.lower().endswith('.pdf'):
                self.pdf_urls.add(full_url)
            elif self.is_valid_internal(full_url):
                found_links.append(full_url)
                
        return found_links

    def prioritize_urls(self, urls: list[str]) -> list[str]:
        """
        Sorts URLs to prioritize keywords matching health insurance policies.
        """
        def score_url(url: str) -> int:
            score = 0
            for kw in self.priority_keywords:
                if kw in url.lower():
                    score += 1
            return score
            
        return sorted(urls, key=score_url, reverse=True)

    def crawl(self) -> dict:
        """
        Executes the domain crawling loop up to max_pages.
        Returns a dictionary containing crawled page contents.
        """
        to_visit = [self.base_url]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        print(f"Starting crawl on domain: {self.domain}")

        while to_visit and len(self.visited_urls) < self.max_pages:
            to_visit = self.prioritize_urls(to_visit)
            url = to_visit.pop(0)

            if url in self.visited_urls:
                continue

            print(f"Crawling: {url}")
            try:
                # Add delay to respect website crawl limits
                time.sleep(0.5)
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    continue

                self.visited_urls.add(url)
                
                # Check content type is HTML
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text or structured HTML segments
                self.crawled_data[url] = response.text
                
                # Discovered children urls
                new_links = self.extract_links(soup, url)
                for link in new_links:
                    if link not in self.visited_urls and link not in to_visit:
                        to_visit.append(link)

            except Exception as e:
                print(f"Failed to crawl {url}: {e}")

        print(f"Crawl complete. Visited {len(self.visited_urls)} pages. Found {len(self.pdf_urls)} PDF links.")
        return {
            "pages": self.crawled_data,
            "pdfs": list(self.pdf_urls)
        }
