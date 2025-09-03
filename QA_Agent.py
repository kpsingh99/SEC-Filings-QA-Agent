import os
import json
import re
import time
from typing import List, Dict, Any
import numpy as np
import logging
import traceback
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



class QueryMetadata(BaseModel):
    """Metadata extracted from a user's financial query."""
    ticker: List[str] = Field(description="List of stock tickers mentioned, e.g., ['AAPL', 'MSFT']")
    sector: List[str] = Field(description="List of industry sectors mentioned, e.g., ['Technology', 'Healthcare']")
    form_type: List[str] = Field(description="List of SEC form types mentioned, e.g., ['10-K', '10-Q']")
    year: List[int] = Field(description="List of four-digit years mentioned, e.g., [2023]")
    search_query: str = Field(description="The core semantic question to be used for the vector search.")

## How sections are defined on the Securities and Exchange Commission, used to find the correct section for a query. 

SEC_SECTIONS = {
    "Item 1. Business": (
        "Provides a general overview of the company's business, including its main products and services, "
        "business model, key markets, acquisitions, competitive positioning, and strategic direction."
    ),
    "Item 1A. Risk Factors": (
        "Describes the most significant risks that could adversely affect the company's business operations, "
        "financial condition, or stock price, helping investors assess potential uncertainties."
    ),
    "Item 2. Properties": (
        "Describes the company's physical assets and properties, including those gained through acquisitions "
        "or business combinations."
    ),
    "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)": (
        "Offers management’s perspective on financial performance, including discussion of revenue, expenses, "
        "liquidity, capital resources, acquisitions, business combinations, restructuring efforts, and overall strategy."
    ),
    "Item 8. Financial Statements and Supplementary Data": (
        "Contains the company’s audited financial statements, including the balance sheet, income statement, "
        "cash flow statement, and notes, which may include details of mergers, acquisitions, and goodwill."
    ),
    "Item 11. Executive Compensation": (
        "Details compensation for the company’s executives, including salaries, bonuses, stock awards, and incentive plans."
    ),
    "8-K Filing (Material Events)": (
        "Reports unscheduled material events or corporate changes, such as mergers and acquisitions, executive departures, "
        "earnings announcements, and other strategic developments. Often includes strategic rationale and financial impact."
    ),
}

# Mapping keywords to specific form types for intelligent filtering 
FORM_TYPE_KEYWORDS = {
    "insider trading": ["3", "4", "5"],
    "executive compensation": ["DEF 14A"],
    "revenue guidance": ["8-K", "10-Q"],
    "financial guidance": ["8-K", "10-Q"],
    "earnings release": ["8-K"],
    "material event": ["8-K"],
    "corporate event": ["8-K"],
    "acquisition": ["8-K"],
    "merger": ["8-K"]
}

class QueryParser:
    """
    An intelligent parser that uses an LLM to extract structured metadata
    from a natural language query and routes it to the most relevant section.
    """
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        
        self.pydantic_parser = PydanticOutputParser(pydantic_object=QueryMetadata)
        self.prompt = PromptTemplate(
            template="""
            You are an expert at parsing financial questions. Analyze the user's query and extract the required information into the specified JSON format.
            User Query: {query}
            Format Instructions: {format_instructions}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.pydantic_parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.pydantic_parser

        self.section_descriptions = list(SEC_SECTIONS.values())
        self.section_names = list(SEC_SECTIONS.keys())
        self.section_embeddings = self.embedding_model.embed_documents(self.section_descriptions)

    def _find_best_section(self, query: str) -> str | None:
        if not query.strip(): return None
        query_embedding = self.embedding_model.embed_query(query)
        similarities = [np.dot(query_embedding, sec_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(sec_emb)) for sec_emb in self.section_embeddings]
        best_index = np.argmax(similarities)
        if similarities[best_index] > 0.75: 
            return self.section_names[best_index]
        return None

    def build_filter(self, query: str) -> (Dict[str, Any], str):
        """Parses a query and builds a ChromaDB filter."""
        parsed_metadata: QueryMetadata = self.chain.invoke({"query": query})
        
        conditions = []
        if parsed_metadata.ticker: conditions.append({"ticker": {"$in": parsed_metadata.ticker}})
        if parsed_metadata.sector: conditions.append({"sector": {"$in": parsed_metadata.sector}})
        
        # Form Type Selection 
        target_forms = set()
        if parsed_metadata.form_type:
            target_forms.update(parsed_metadata.form_type)
        else:
            found_keyword = False
            for keyword, forms in FORM_TYPE_KEYWORDS.items():
                if keyword in parsed_metadata.search_query.lower():
                    target_forms.update(forms)
                    found_keyword = True
            if not found_keyword:
                target_forms.add("10-K")
        
        conditions.append({"form_type": {"$in": list(target_forms)}})

        if parsed_metadata.year:
            conditions.append({"year": {"$in": parsed_metadata.year}})
        
        # Only search for "Item" sections if we are looking at a 10-K or 10-Q
        if any(form in target_forms for form in ["10-K", "10-Q"]):
            best_section = self._find_best_section(parsed_metadata.search_query)
            if best_section:
                simple_section_match = re.match(r"(?i)Item\s\d{1,2}[A-Z]?", best_section)
                if simple_section_match:
                    simple_section_tag = simple_section_match.group(0).upper().replace(" ", "")
                    conditions.append({"section_simple": simple_section_tag})
        
            
        filter_dict = {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else {})
        return filter_dict, parsed_metadata.search_query


# QA Implementation 
def pretty_print_result(result: dict):
    """Prints the QA result in a readable format."""
    print("\n✅ Final Answer:")
    print(result["result"])
    print("\n" + "---" * 10)
    print("📚 Source Documents Used:")
    
    printed_sources = set()
    for source in result["source_documents"]:
        source_id = (
            source.metadata.get('filename'), 
            source.metadata.get('section_full') 
        )
        if source_id not in printed_sources:
            print(f"  - File: {source.metadata.get('filename')}\n    Section: {source.metadata.get('section_full', 'N/A')}")
            printed_sources.add(source_id)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY environment variable not set.")
    else:
        # 1. Load components
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_directory = "sec_filings_db2"
        
        if not os.path.exists(persist_directory):
            print(f"❌ Error: Vector store not found at '{persist_directory}'. Please run the processing script first.")
        else:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
            
            # 2. Initialize the intelligent query parser
            parser = QueryParser(llm, embedding_model)

            # 3. Define the user's question
	
            user_question = "How do companies describe competitive advantages? What themes emerge?"
            
            
            ## If question is not computable. 
            if not user_question or not isinstance(user_question, str):
                raise ValueError("User question must be a non-empty string")
            
            # 4. Use the parser to build the filter and get the core query
            try:
                metadata_filter, search_query = parser.build_filter(user_question)
                
                print(f"❓ User Question: {user_question}")
                print(f"🔍 Search Query: {search_query}")
                print(f"⚙️ Generated Filter: {json.dumps(metadata_filter, indent=2)}")

                retriever = vector_store.as_retriever(
					search_kwargs={"k": 8, "filter": metadata_filter},
					search_type="mmr"  
				)
                retrieved_docs = retriever.get_relevant_documents(search_query)
                if not retrieved_docs:
                    print(" No relevant documents found for the query.")
                # 6. Use the simple and fast "stuff" chain with a custom prompt
                prompt_template = """
                You are a senior financial analyst. Your task is to synthesize a high-level answer to the user's question based on the provided context from multiple SEC filings.

                **Instructions:**
                1.  Read all the provided context carefully. Each document is from a different source.
                2.  Identify the main themes, patterns, and key differences that emerge across the documents.
                3.  Structure your answer clearly. Start with a high-level summary, then provide a bulleted list of the common themes.
                4.  Do not just list the information from each source. Your value is in the synthesis and comparison.
                5.  If the context is insufficient to answer the question, state that clearly.

                **Context from documents:**
                {context}

                **User's Question:**
                {question}

                **Final Synthesized Answer:**
                """
                
                QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )
                
                result = qa_chain.invoke({"query": search_query})
                
                pretty_print_result(result)
            except ValueError as ve:
                print(f"❌ Invalid input: {ve}")
            except RuntimeError as re:
                print(f"❌ Runtime error during processing: {re}")
            except Exception as e:
            	print(f"❌ Unexpected error: {e}", traceback.format_exc())
