# RAG Advanced

A Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent question-answering capabilities using embeddings, vector search, and large language models.

## Features

- **PDF Processing**: Extract and chunk text from PDF documents using intelligent splitting strategies
- **Vector Embeddings**: Generate semantic embeddings using SentenceTransformers
- **Vector Storage**: Store and query embeddings using ChromaDB
- **Query Augmentation**: Enhance queries with hypothetical answers for improved retrieval
- **LLM Integration**: Generate responses using Mistral AI models
- **Visualization**: Plot embedding spaces with UMAP dimensionality reduction

## Requirements

- Python >=3.13
- Mistral API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-advanced
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

## Usage

### Basic Usage

Run the main RAG pipeline:
```bash
python main.py
```

This will:
1. Process the NVIDIA annual report PDF in `data/`
2. Split text into chunks and generate embeddings
3. Store embeddings in ChromaDB
4. Execute a sample query with query augmentation
5. Generate and display the answer
6. Create a visualization of the embedding space

### Customizing Queries

Edit the `original_query` variable in `main.py:112` to ask different questions:
```python
original_query = "What was the total profit for the year, and how does it compare to the previous year?"
```

### Adding Your Own Documents

1. Place PDF files in the `data/` directory
2. Update the file path in `main.py:42`:
```python
reader = PdfReader("data/your-document.pdf")
```

## Project Structure

```
├── data/                           # PDF documents for processing
│   └── NVIDIA-2025-Annual-Report.pdf
├── main.py                         # Main RAG pipeline
├── helper_utils.py                 # Utility functions
├── .env.example                    # Environment variables template
├── .python-version                 # Python version specification
├── AGENTS.md                       # Guidelines for AI coding agents
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Dependency lock file
└── README.md                       # This file
```

## How It Works

1. **Document Processing**: PDFs are loaded and text is extracted
2. **Text Chunking**: Documents are split into manageable chunks using recursive character splitting and token-based splitting
3. **Embedding Generation**: Each chunk is converted to vector embeddings using SentenceTransformers
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
5. **Query Augmentation**: Original queries are enhanced with hypothetical answers to improve retrieval
6. **Retrieval**: Most relevant document chunks are retrieved using similarity search
7. **Answer Generation**: Mistral AI generates contextual answers based on retrieved chunks
8. **Visualization**: UMAP creates 2D projections of the embedding space for analysis

## Environment Variables

- `MISTRAL_API_KEY`: Your Mistral AI API key (required)
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid tokenizer warnings

## Troubleshooting

- Ensure your Mistral API key is valid and has sufficient credits
- Check that PDF files are readable and not encrypted
- Verify Python version compatibility (>=3.13 required)
- If you encounter import errors, run `uv sync` to install all dependencies
- For tokenizer warnings, set `TOKENIZERS_PARALLELISM=false` in your environment