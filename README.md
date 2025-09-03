# SEC Filings QA Agent

## Overview

This project is a sophisticated Question-Answering (QA) agent designed to analyze and answer complex financial research questions based on SEC filings. It leverages a Retrieval-Augmented Generation (RAG) pipeline, combining a powerful language model with a vector database of SEC filings to provide accurate, context-aware, and source-attributed answers.

The system is capable of processing thousands of filings from multiple companies, preserving document structure and table data, and intelligently parsing natural language queries to perform precise, metadata-driven searches.

## Features

- Automated Data Processing: Scripts to download and process thousands of SEC filings into a clean, searchable format.

- Intelligent Text & Table Extraction: Robustly parses messy HTML, converts complex tables to structured JSON, and preserves surrounding context.

- Intelligent Query Parser: Uses an LLM to automatically extract tickers, form types, years, and sectors from user questions to create precise search filters.

- Semantic Section Routing: Automatically identifies the most relevant section of a filing (e.g., "Item 1A. Risk Factors") for a given query.

- Advanced Retrieval: Employs Maximal Marginal Relevance (MMR) search to ensure diverse and relevant results for complex comparative questions.

- Source Attribution: All answers are accompanied by a list of the source documents used for generation.

## Setup and Installation

### 1. Prerequisites

- Python 3.9+
- An API key from a data source for SEC filings (e.g., sec-api.io)
- A Google AI API key for the Gemini model

### 2. Environment Setup

It is highly recommended to use a virtual environment to manage project dependencies.

# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


### 3. Setup libraries 

pip install langchain langchain-google-genai langchain-community chromadb beautifulsoup4 numpy pydantic requests

```bash


## How to run the Agent.
Just run the SEC_Agent.ipynb file cell by cell. 

Put your question in user_question in the cell (asked) . 
