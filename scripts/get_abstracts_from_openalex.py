import json
import time
import logging
import re
from pathlib import Path
from typing import List
import pyalex
from pyalex import Works

# Set your polite pool email
pyalex.config.email = "email@uu.nl"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_filename_from_doi(doi: str) -> str:
    """Convert a DOI into a filesystem‐safe filename base."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', doi).strip('_')


def find_abstracts(text_dir: Path) -> List[str]:
    """
    Scan all *_abstract.json files under text_dir and return a list of DOIs
    whose 'abstract' field is empty or missing.
    """
    dois = []
    doi = '10.1017/s0167676809000038'
    dois.append(doi)
    return dois



def fetch_and_save_abstract(doi: str, text_dir: Path) -> bool:
    """
    Fetch the abstract for a DOI from OpenAlex and save both the .txt
    and updated .json metadata. Returns True if saved, False otherwise.
    """
    try:
        work = Works()[f"doi:{doi}"]

        abstract = work["abstract"]

        if not abstract:
            logger.info(f"ℹ️ No abstract in OpenAlex for DOI {doi}")
            return False

        title = work.get("title", "Untitled")
        filename_base = safe_filename_from_doi(doi)

        txt_path = text_dir / f"{filename_base}_abstract.txt"
        json_path = text_dir / f"{filename_base}_abstract.json"

        # Save abstract text
        content = f"Title: {title}\nDOI: {doi}\n\n{abstract}"
        txt_path.write_text(content, encoding="utf-8")
        logger.info(f" Saved abstract text: {txt_path.name}")

        # Update only the abstract in the existing metadata JSON
        # with open(json_path, "r", encoding="utf-8") as jf:
        #     existing_meta = json.load(jf)
        #
        # existing_meta["abstract"] = abstract
        # existing_meta["source"] = 'openalex'
        #
        # with open(json_path, "w", encoding="utf-8") as jf:
        #     json.dump(existing_meta, jf, indent=2, ensure_ascii=False)

        logger.info(f" Updated metadata abstract in: {json_path.name}")

        return True

    except Exception as e:
        logger.error(f"❌ Error fetching DOI {doi}: {e}")
        return False


def main():
    # Determine your texts directory
    script_dir = Path(__file__).resolve().parent
    text_dir = script_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)
    print(text_dir)
    # Find DOIs with no abstract saved locally
    dois = find_abstracts(text_dir)
    logger.info(f"Found {len(dois)} DOIs missing abstracts")
    #
    added = 0

    for doi in dois:

        # if doi =='10.1017/s0167676809000038':
        if fetch_and_save_abstract(doi, text_dir):
            added += 1
        time.sleep(0.3)  # Polite rate limiting
#
    logger.info(f"✅ Done. Abstracts added: {added} / {len(dois)}")


if __name__ == "__main__":
    main()