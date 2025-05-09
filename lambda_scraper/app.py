# app.py
import requests
from bs4 import BeautifulSoup
import json

# --- Core Scraping Logic ---
def fetch_html(url):
    """Fetches HTML content from a given URL."""
    try:
        headers = { # Basic user-agent to mimic a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10) # 10-second timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def parse_content(html_content, selectors=None):
    """
    Parses HTML content using BeautifulSoup to extract data based on CSS selectors.
    If selectors is None, it will attempt a generic extraction (e.g., all paragraph text).
    
    Supports custom selector syntax:
    - selector::text - Gets text content from matching elements
    - selector::attr(attribute_name) - Gets specified attribute from matching elements
    """
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    
    extracted_data = {}

    if selectors:
        # Example: selectors = {"title": "h1.main-title", "paragraphs": "div.content p"}
        for key, selector in selectors.items():
            # Process custom selector syntax BEFORE passing to BeautifulSoup
            if "::text" in selector:
                # Extract the actual CSS selector without the ::text part
                actual_selector = selector.split("::text")[0]
                elements = soup.select(actual_selector)
                extracted_data[key] = [el.get_text(separator=' ', strip=True) for el in elements]
                
            elif "::attr(" in selector:
                # Extract attribute name and the actual selector
                try:
                    attr_name = selector.split("::attr(")[1].split(")")[0]  # extracts attribute name
                    actual_selector = selector.split("::attr(")[0]
                    elements = soup.select(actual_selector)
                    extracted_data[key] = [el.get(attr_name) for el in elements if el.has_attr(attr_name)]
                except IndexError:
                    # Handle malformed attr selector
                    print(f"Warning: Malformed attribute selector: {selector}")
                    extracted_data[key] = []
                
            else:
                # Regular selector without custom syntax
                elements = soup.select(selector)
                extracted_data[key] = [el.get_text(separator=' ', strip=True) for el in elements]

    else:
        # Generic extraction: all paragraphs and the title
        title = soup.find('title')
        extracted_data['title'] = title.get_text(strip=True) if title else "No title found"
        
        paragraphs = soup.find_all('p')
        extracted_data['paragraphs'] = [p.get_text(separator=' ', strip=True) for p in paragraphs]
        
        links = soup.find_all('a', href=True)
        extracted_data['links'] = [link['href'] for link in links[:20]] # First 20 links

    return extracted_data

# --- Lambda Handler ---
def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    Expected event:
    {
        "url": "http://example.com",
        "selectors": {  // Optional
            "main_heading": "h1.some-class::text",
            "article_body": "div.article p::text",
            "image_srcs": "img.important-image::attr(src)"
        }
    }
    """
    print(f"Received event: {json.dumps(event)}")

    url = event.get('url')
    selectors = event.get('selectors') # This can be None

    if not url:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'URL is required'})
        }

    html_content = fetch_html(url)
    if not html_content:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Failed to fetch content from {url}'})
        }

    scraped_data = parse_content(html_content, selectors)
    if not scraped_data:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Failed to parse content from {url}'})
        }

    return {
        'statusCode': 200,
        'body': json.dumps(scraped_data)
    }

# --- Local Testing (Optional) ---
if __name__ == '__main__':
    # Test with a known URL (e.g., a news site, or http://books.toscrape.com/)
    # Ensure the website allows scraping in its robots.txt
    test_event_generic = {'url': 'http://books.toscrape.com/'}
    test_event_specific_selectors = {
        'url': 'http://books.toscrape.com/',
        'selectors': {
            'book_titles': 'article.product_pod h3 a::attr(title)', # Gets the title attribute
            'prices': 'article.product_pod .price_color::text' # Gets the text
        }
    }

    print("--- Generic Scraping Test ---")
    result_generic = lambda_handler(test_event_generic, None)
    print(json.dumps(json.loads(result_generic['body']), indent=2))
    
    print("\n--- Specific Selector Test ---")
    result_specific = lambda_handler(test_event_specific_selectors, None)
    print(json.dumps(json.loads(result_specific['body']), indent=2))