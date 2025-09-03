import os
import json
import re
import shutil
from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_html_table_to_json(table_soup: BeautifulSoup) -> str:
    """
    Converts a complex HTML table with rowspan and colspan into a structured JSON string.
    This version is updated to be more robust against malformed tables.
    """
    rows = table_soup.find_all('tr')
    
    max_cols = 0
    for row in rows:
        max_cols = max(max_cols, len(row.find_all(['th', 'td'])))
    
    # Create a virtual grid with a safety buffer
    grid = [[None for _ in range(max_cols + 20)] for _ in range(len(rows))]

    # Populate the grid, accounting for rowspan and colspan
    for r, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        c = 0
        for cell in cells:
            while c < len(grid[r]) and grid[r][c] is not None:
                c += 1
            
            if c >= len(grid[r]):
                continue # Skip cell if row is malformed and we're out of bounds
            
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            
            cell_text = cell.get_text(strip=True)
            
            for i in range(rowspan):
                for j in range(colspan):
                    # --- FIX: Add boundary checks for rowspan/colspan ---
                    if r + i < len(grid) and c + j < len(grid[r + i]):
                        grid[r + i][c + j] = cell_text
                    # ----------------------------------------------------
            c += colspan

    # Find the header row (first row with significant content)
    header = []
    header_row_index = -1
    for r, row_data in enumerate(grid):
        if any(cell and isinstance(cell, str) for cell in row_data):
            # Clean up header by removing empty trailing columns
            last_content_col = -1
            for c, cell in enumerate(row_data):
                if cell is not None:
                    last_content_col = c
            header = [cell if cell is not None else '' for cell in row_data[:last_content_col+1]]
            header_row_index = r
            break

    if not header:
        return "" # return if there is no header. 

    # Grid -> Dictionarise 
    json_data = []
    for r in range(header_row_index + 1, len(grid)):
        row_data = grid[r]
        if any(cell is not None for cell in row_data):
            row_dict = {header[c]: cell for c, cell in enumerate(row_data) if c < len(header)}
            if any(row_dict.values()):
                json_data.append(row_dict)

    return json.dumps(json_data, indent=2) if json_data else ""


def extract_text_and_tables(html: str) -> str:
    """
    Cleans HTML, converts tables to JSON, retains surrounding context for tables,
    and returns a single combined text for processing.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Process and replace all tables
    for table in soup.find_all('table'):
        # a. Retain surrounding context from the preceding paragraph
        context_text = ""
        prev_p = table.find_previous_sibling()
        if prev_p and prev_p.name == 'p':
            context_text = prev_p.get_text(strip=True)

        # b. Capture table caption (if available)
        caption = table.find('caption')
        caption_text = caption.get_text(strip=True) if caption else ""

        # c. Capture nearest preceding heading (if available)
        heading = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        heading_text = heading.get_text(strip=True) if heading else ""

        # d. Convert table to JSON
        json_table = convert_html_table_to_json(table)

        # e. Create a combined block with all context
        combined_block = f"\n\n--- TABLE START ---\n"
        if heading_text:
            combined_block += f"Heading: {heading_text}\n"
        if caption_text:
            combined_block += f"Caption: {caption_text}\n"
        if context_text:
            combined_block += f"Context: {context_text}\n"
        if json_table:
            combined_block += f"Table Data (JSON):\n{json_table}\n"
        combined_block += "--- TABLE END ---\n\n"
        
        # f. Replace the original table with the combined block
        table.replace_with(BeautifulSoup(combined_block, "html.parser"))

    # 2. Clean remaining non-content tags
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    for tag in soup(re.compile(r".*:[a-zA-Z]")):
        tag.decompose()
    
    # 3. Extract all text, which now includes the contextualized JSON tables
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text

def extract_sections(text: str) -> list[tuple[str, str]]:
    """Extracts (section_title, section_content) tuples from full filing text."""
    pattern = re.compile(
        r"(?i)(Item\s\d{1,2}[A-Z]?(?:[.:–\-]?\s?.*?))(?=\nItem\s\d{1,2}[A-Z]?(?:[.:–\-]|\s)|\Z)", 
        re.DOTALL
    )
    matches = pattern.findall(text)
    results = []
    for match in matches:
        split_point = match.find("\n")
        title = match[:split_point].strip() if split_point != -1 else match.strip()
        content = match[split_point:].strip() if split_point != -1 else ""
        title = re.sub(r"\s+", " ", title)
        if content:
            results.append((title, content))
    return results

def process_company_filings(ticker: str, sector: str, base_folder: str = ".") -> list[Document]:
    """Processes all downloaded HTML filings for a given company and returns a list of Document chunks."""
    folder_path = os.path.join(base_folder, f"../{ticker}_sec_filings")
    metadata_path = os.path.join(folder_path, f"{ticker}_metadata.json")

    if not os.path.exists(folder_path) or not os.path.exists(metadata_path):
        logging.warning(f"Folder or metadata file not found for ticker {ticker}. Skipping.")
        return []

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    metadata_lookup = {item['filename']: item for item in metadata_list}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    company_chunks = []

    for filename, file_metadata in metadata_lookup.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Missing file, skipping: {filename}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()

            text = extract_text_and_tables(html)
            print("Text -> ", text)
            sections = extract_sections(text)

            if text and sections:
                first_item_start = text.find(sections[0][0])
                if first_item_start > 0:
                    preamble_content = text[:first_item_start].strip()
                    if preamble_content:
                        sections.insert(0, ("Preamble", preamble_content))
            elif text and not sections:
                sections = [("Full Document", text)]

            for title, content in sections:
                if not content.strip():
                    continue
                
                metadata = file_metadata.copy()
                metadata["section_full"] = title 
                metadata["sector"] = sector
                metadata["contains_table"] = "--- TABLE START ---" in content
                
                try:
                    metadata["year"] = int(file_metadata["filed_at"][:4])
                except (ValueError, TypeError):
                    metadata["year"] = 0
                
                simple_section_match = re.match(r"(?i)Item\s\d{1,2}[A-Z]?", title)
                metadata["section_simple"] = simple_section_match.group(0).upper().replace(" ", "") if simple_section_match else "OTHER"
                
                chunks = text_splitter.create_documents([content], metadatas=[metadata])
                company_chunks.extend(chunks)

            logging.info(f"✅ Processed {filename}")

        except Exception as e:
            logging.error(f"❌ Error processing {filename}: {e}", exc_info=True)

    return company_chunks

def filter_complex_metadata(metadata: dict) -> dict:
    """Filters metadata to ensure all values are simple types that ChromaDB can handle."""
    filtered_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        elif isinstance(value, list):
            filtered_metadata[key] = ", ".join(map(str, value))
    return filtered_metadata

if __name__ == "__main__":
    
    TICKER_TO_SECTOR = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "JPM": "Financial Services", "V": "Financial Services",
        "JNJ": "Healthcare", "PFE": "Healthcare",
        "BA": "Industrials", "CAT": "Industrials", "UPS": "Industrials",
        "XOM": "Energy", "GOOG": "Technology", "UNH": "Healthcare",
        "PG": "Consumer Staples", "HD": "Consumer Discretionary",
        "CVX": "Energy", "MRK": "Healthcare", "BAC": "Financial Services"
    }
    tickers = list(TICKER_TO_SECTOR.keys())
    
    all_final_chunks = []
    for ticker in tickers:
        logging.info(f"--- Processing ticker: {ticker} ---")
        sector = TICKER_TO_SECTOR.get(ticker, "Unknown")
        ticker_chunks = process_company_filings(ticker, sector, base_folder=".")
        all_final_chunks.extend(ticker_chunks)
        logging.info(f"--- Finished {ticker}. Chunks so far: {len(all_final_chunks)} ---")

    cleaned_chunks = [
        Document(page_content=doc.page_content, metadata=filter_complex_metadata(doc.metadata))
        for doc in all_final_chunks
    ]

    logging.info(f"📦 Total chunks created for all companies: {len(cleaned_chunks)}")

    if cleaned_chunks:
        logging.info("--- Creating and saving ChromaDB vector store ---")
        if not os.getenv("GOOGLE_API_KEY"):
            logging.warning("⚠️ GOOGLE_API_KEY environment variable not set.")
        else:
            try:
                embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                persist_directory = "sec_filings_db"
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory)
                vector_store = Chroma.from_documents(
                    documents=cleaned_chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory
                )
                logging.info(f"✅ ChromaDB vector store created and saved to '{persist_directory}'")
            except Exception as e:
                logging.error(f"❌ Failed to create vector store: {e}", exc_info=True)
