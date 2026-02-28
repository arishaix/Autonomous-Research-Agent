from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def perform_search(query: str, max_results: int = 2) -> List[Dict[str, str]]:
    """
    Searches DuckDuckGo and returns a list of dictionaries with 'title', 'url', and 'snippet'.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"Error searching for {query}: {e}")
    return results

def scrape_webpage(url: str) -> str:
    """
    Fetches the webpage HTML and extracts the main text content, stripping tags.
    Returns a truncated string to save LLM context window space.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove noisy elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        # Truncate to 4000 characters to keep our prompt size manageable
        return text[:4000]
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""
