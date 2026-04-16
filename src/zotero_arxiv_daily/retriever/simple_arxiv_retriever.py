from datetime import datetime, timedelta
import requests
from datetime import datetime, timedelta
import feedparser
from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..paper import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
from loguru import logger


PDF_EXTRACT_TIMEOUT = 180


class SimpleArxivRetriever():
    def __init__(self, config):
        self.config = config
    
    def retrieve_papers(self) -> list[Paper]:
        
        if self.config.executor.get('from_yesterday', False):
            print(f'get papers from yesterday ...')
            papers = get_yesterday_papers()
        else:
            papers = self.get_rss_papers()

        if self.config.executor.debug:
            papers = papers[:10]
        return papers

    def get_rss_papers(self):
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        
        papers = []
        for entry in feed.entries:
            if not entry.get("arxiv_announce_type", "new") in allowed_announce_types:
                continue

            p = Paper(
                source='arxiv',
                title=entry.title,
                authors=[a.name for a in entry.authors],
                abstract=entry.summary.split('Abstract:')[-1],
                url=entry.link,
                pdf_url=entry.link.replace('abs', 'pdf'),
                full_text=None
            )
            papers.append(p)

        return papers

    

def get_yesterday_papers():
    # Define the categories and date range
    categories = ["cs.AI", "cs.CV", "cs.LG", "cs.CL"]
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the search query for the arXiv API
    search_query = " OR ".join([f"cat:{cat}" for cat in categories])
    base_url = "http://export.arxiv.org/api/query?"

    params = {
        "search_query": f"({search_query}) AND lastUpdatedDate:[{yesterday}0000 TO {yesterday}2359]",
        "start": 0,
        "max_results": 1000,  # Adjust as needed
    }

    # Make the API request
    response = requests.get(base_url, params=params)
    # print(response.text)

    # Parse the XML response
    feed = feedparser.parse(response.text)
    print(f'retrieved {len(feed.entries)}')
    raw_papers = feed.entries
    for paper in raw_papers:
        for link in paper['links']:
            if link.get('type') == 'application/pdf':
                paper['pdf_url'] = link['href']
                break

    papers = []
    for entry in feed.entries:
        p = Paper(
            source='arxiv',
            title=entry.title,
            authors=[a.name for a in entry.authors],
            abstract=entry.summary,
            url=entry.link,
            pdf_url=entry.pdf_url,
            full_text=None
        )
        papers.append(p)

    return papers

