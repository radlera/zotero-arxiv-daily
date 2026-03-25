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


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10,delay_seconds=10)
        
        if self.config.executor.get('from_yesterday', False):
            print(f'get papers from yesterday ...')
            all_paper_ids = get_yesterday_papers()
        else: 
            all_paper_ids = self.get_rss_papers(client)

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        raw_papers = []
        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper:ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        # try:
        #     with ThreadPoolExecutor(max_workers=1) as pool:
        #         full_text = pool.submit(extract_text_from_pdf, raw_paper).result(timeout=PDF_EXTRACT_TIMEOUT)
        # except TimeoutError:
        #     logger.warning(f"PDF extraction timed out for {raw_paper.title}")
        #     full_text = None
            
        # if full_text is None:
        #     full_text = extract_text_from_tar(raw_paper)
        full_text = None
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )
    
    def get_rss_papers(self, client):
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        return all_paper_ids

def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None
        urlretrieve(paper.pdf_url, path)
        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None
        return full_text

def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None
        urlretrieve(source_url, path)
        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {paper.title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None
        return full_text
    

def get_yesterday_papers():
    # Define the categories and date range
    categories = ["cs.AI", "cs.CV", "cs.LG", "cs.CL"]
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Construct the search query for the arXiv API
    search_query = " OR ".join([f"cat:{cat}" for cat in categories])
    base_url = "http://export.arxiv.org/api/query?"

    params = {
        "search_query": f"({search_query}) AND submittedDate:[{yesterday}0000 TO {yesterday}2359]",
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
        paper['entry_id'] = paper['link']

    all_paper_ids = [
            i.id.removeprefix("http://arxiv.org/abs/").split('v')[0]
            for i in raw_papers
        ]
    
    print(f'fetched {len(all_paper_ids)} papers')

    return all_paper_ids

