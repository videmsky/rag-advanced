# Agent Guidelines for rag-advanced

## Build/Test/Lint Commands
- **Run main script**: `python main.py`
- **Test execution**: No formal test framework - validate by running main pipeline
- **No linting configured**: Standard Python style guidelines apply
- **Install dependencies**: `uv sync` or `pip install -e .`
- **Environment setup**: Copy `.env.example` to `.env` and add `MISTRAL_API_KEY`

## Code Style Guidelines
- **Python version**: Requires Python >=3.13
- **Import style**: Group imports (stdlib, third-party, local) with line breaks between groups
- **Function definitions**: Use descriptive docstrings with Args/Returns sections  
- **Type hints**: Use type comments (e.g., `# type: ignore`) when needed for complex types
- **Logging**: Use timestamped print statements with status indicators (✓, ⚙) for progress tracking
- **Naming**: snake_case for functions/variables, descriptive names preferred
- **String formatting**: Use f-strings for string interpolation
- **Error handling**: Use `raise ValueError()` with descriptive messages for validation
- **Line length**: Approximately 87 characters (based on word_wrap utility)
- **Indentation**: 2-space indentation for function arguments/parameters

## Project Structure
- **Main entry**: `main.py` - RAG pipeline with PDF processing and embeddings
- **Utilities**: `helper_utils.py` - Helper functions for embeddings and text processing
- **Data**: `data/` directory contains PDF documents for processing
- **Environment**: Uses `.env` file for API keys (MISTRAL_API_KEY required)

## Key Dependencies
- ChromaDB for vector storage, LangChain for text splitting, Mistral for LLM, sentence-transformers for embeddings