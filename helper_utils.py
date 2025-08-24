# helper_utils.py
import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader
from datetime import datetime


def project_embeddings(embeddings, umap_transform):
  """
  Projects the given embeddings using the provided UMAP transformer.

  Args:
  embeddings (numpy.ndarray): The embeddings to project.
  umap_transform (umap.UMAP): The trained UMAP transformer.

  Returns:
  numpy.ndarray: The projected embeddings.
  """
  projected_embeddings = umap_transform.transform(embeddings)
  return projected_embeddings


def word_wrap(text, width=87):
  """
  Wraps the given text to the specified width.

  Args:
  text (str): The text to wrap.
  width (int): The width to wrap the text to.

  Returns:
  str: The wrapped text.
  """
  return "\n".join([text[i : i + width] for i in range(0, len(text), width)])


def extract_text_from_pdf(file_path):
  """
  Extracts text from a PDF file.

  Args:
  file_path (str): The path to the PDF file.

  Returns:
  str: The extracted text.
  """
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Extracting text from PDF: {file_path}")
  text = []
  with open(file_path, "rb") as f:
    pdf = PdfReader(f)
    total_pages = len(pdf.pages)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ PDF loaded: {total_pages} pages found")
    
    for page_num in range(total_pages):
      page = pdf.pages[page_num]
      text.append(page.extract_text())
      if (page_num + 1) % 10 == 0 or page_num + 1 == total_pages:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Processed {page_num + 1}/{total_pages} pages")
  
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Text extraction completed")
  return "\n".join(text)


def load_chroma(filename, collection_name, embedding_function):
  """
  Loads a document from a PDF, extracts text, generates embeddings, and stores it in a Chroma collection.

  Args:
  filename (str): The path to the PDF file.
  collection_name (str): The name of the Chroma collection.
  embedding_function (callable): A function to generate embeddings.

  Returns:
  chroma.Collection: The Chroma collection with the document embeddings.
  """
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Loading document into Chroma collection: {collection_name}")
  
  # Extract text from the PDF
  text = extract_text_from_pdf(filename)

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Splitting text into paragraphs...")
  # Split text into paragraphs or chunks
  paragraphs = text.split("\n\n")
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Created {len(paragraphs)} text chunks")

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Generating embeddings for text chunks...")
  # Generate embeddings for each chunk
  embeddings = []
  for i, paragraph in enumerate(paragraphs):
    embeddings.append(embedding_function(paragraph))
    if (i + 1) % 50 == 0 or i + 1 == len(paragraphs):
      print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Generated embeddings for {i + 1}/{len(paragraphs)} chunks")

  # Create a DataFrame to store text and embeddings
  data = {"text": paragraphs, "embeddings": embeddings}
  df = pd.DataFrame(data)

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Creating Chroma collection...")
  # Create or load the Chroma collection
  collection = chromadb.Client().create_collection(collection_name)

  print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Adding data to Chroma collection...")
  # Add the data to the Chroma collection
  for idx, row in df.iterrows():
    collection.add(ids=[str(idx)], documents=[row["text"]], embeddings=[row["embeddings"]])
    if (idx + 1) % 25 == 0 or idx + 1 == len(df):
      print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙ Added {idx + 1}/{len(df)} documents to collection")
  
  print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Chroma collection loaded successfully with {len(df)} documents")
  return collection