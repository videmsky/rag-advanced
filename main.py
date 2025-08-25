from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from mistralai import Mistral
from dotenv import load_dotenv
# UMAP for dimensionality reduction and visualization
import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# Matplotlib for plotting embeddings visualization
import matplotlib.pyplot as plt
from datetime import datetime

# LangChain text splitters for document chunking
from langchain.text_splitter import (
  RecursiveCharacterTextSplitter,
  SentenceTransformersTokenTextSplitter,
)

def main():
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loading environment variables...")
  # Load environment variables from .env file
  load_dotenv()

  # Get Mistral API key from environment variables
  mistral_api_key = os.getenv("MISTRAL_API_KEY")
  if mistral_api_key is None:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Initializing Mistral client...")
  # Initialize Mistral client with API key
  client = Mistral(api_key=mistral_api_key)

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Loading PDF document...")
  # Read PDF document and extract text from all pages
  reader = PdfReader("data/NVIDIA-2025-Annual-Report.pdf")
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Loading PDF: {len(reader.pages)} pages found")
  
  pdf_texts = [p.extract_text().strip() for p in reader.pages]

  # Filter out empty strings from extracted text
  pdf_texts = [text for text in pdf_texts if text]
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Extracted text from {len(pdf_texts)} pages")

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Splitting text into chunks...")
  # Split text into smaller chunks using character-based splitter
  # Uses hierarchical separators: paragraphs -> sentences -> words -> characters
  character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
  )
  character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

  # Further split chunks based on token count for embedding model compatibility
  token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
  )
  token_split_texts = []
  for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Split documents into {len(token_split_texts)} chunks")

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Setting up vector database...")
  # Initialize sentence transformer embedding function
  embedding_function = SentenceTransformerEmbeddingFunction()

  # Create ChromaDB client and collection for vector storage
  chroma_client = chromadb.Client()
  chroma_collection = chroma_client.create_collection(
    "annual-report-collection", embedding_function=embedding_function # type: ignore
  )

  # Generate embeddings for all text chunks and store in vector database
  ids = [str(i) for i in range(len(token_split_texts))]
  chroma_collection.add(ids=ids, documents=token_split_texts)
  chroma_collection.count()
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Stored {len(token_split_texts)} document chunks in vector database")

  # Sample query for testing (not used in main pipeline)
  # query = "What was the total revenue for the year?"

  # Function to generate hypothetical answers for query augmentation
  def augment_query_generated(query, model="ministral-8b-latest"):
    prompt = """You are a helpful expert financial research assistant. 
    Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
      {
        "role": "system",
        "content": prompt,
      },
      {
        "role": "user",
        "content": query
      },
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Making API call to Mistral for query augmentation...")
    # Generate hypothetical answer using Mistral model
    response = client.chat.complete(
      model=model,
      messages=messages,
    )
    content = response.choices[0].message.content
    return content
  
  # Main query to be answered by the RAG system
  original_query = "What was the total profit for the year, and how does it compare to the previous year?"
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Generating hypothetical answer for query enhancement...")
  # Generate hypothetical answer to improve retrieval (HyDE technique)
  hypothetical_answer = augment_query_generated(original_query)
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Query enhanced with hypothetical answer")

  # Combine original query with hypothetical answer for better semantic search
  joint_query = f"{original_query} {hypothetical_answer}"

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Searching for relevant documents...")
  # Query vector database with augmented query to retrieve relevant documents
  results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
  )
  retrieved_documents = results["documents"][0] # type: ignore
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Retrieved {len(retrieved_documents)} relevant document chunks")

  # Function to generate final answer using retrieved context
  def generate_response(question, relevant_chunks):
    # Combine retrieved chunks into single context
    context = "\n\n".join(relevant_chunks)
    prompt = (
      "You are an assistant for question-answering tasks. Use the following pieces of "
      "retrieved context to answer the question. If you don't know the answer, say that you "
      "don't know. Be detailed in your answer. "
      "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Making API call to Mistral for final answer generation...")
    # Generate final answer using context and question
    response = client.chat.complete(
      model="ministral-8b-latest",
      messages=[
        {
          "role": "system",
          "content": prompt,
        },
        {
          "role": "user",
          "content": question,
        },
      ],
    )
    answer = response.choices[0].message.content
    return answer
  
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Generating final answer...")
  # Generate and display the final answer
  answer = generate_response(joint_query, retrieved_documents)
  
  print("==== Question ====")
  print(f"Oringal Query: {original_query}")
  print("==== Hypothetical Answer ====")
  print(hypothetical_answer)
  print("==== Actual Answer ====")
  print("#######################")
  print(answer)
  
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Preparing embeddings visualization...")
  # Get all embeddings from the collection for visualization
  embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Fitting UMAP transformer...")
  # Fit UMAP transformer for dimensionality reduction to 2D
  umap_transform = umap.UMAP(transform_seed=0).fit(embeddings)
  # Project all document embeddings to 2D space
  projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

  # Get embeddings for retrieved documents and queries
  retrieved_embeddings = results["embeddings"][0] # type: ignore
  original_query_embedding = embedding_function([original_query])
  augmented_query_embedding = embedding_function([joint_query])

  # Project query and retrieved document embeddings to 2D space
  projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
  )
  projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
  )
  projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
  )

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Creating visualization plot...")
  # Create visualization of embeddings in 2D space
  plt.figure()

  # Plot all document embeddings as gray dots
  plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
    label="All Documents"
  )
  # Highlight retrieved documents with green circles
  plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
    label="Retrieved Documents"
  )
  # Mark original query with red X
  plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
    label="Original Query"
  )
  # Mark augmented query with orange X
  plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
    label="Augmented Query"
  )

  # Add legend to the plot
  plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
  
  # Configure plot appearance and display
  plt.axis("equal")
  plt.title(f"{original_query}")
  plt.show()  # display the plot

if __name__ == "__main__":
  main()
