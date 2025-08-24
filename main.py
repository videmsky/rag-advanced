from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from mistralai import Mistral
from dotenv import load_dotenv
import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib.pyplot as plt

from langchain.text_splitter import (
  RecursiveCharacterTextSplitter,
  SentenceTransformersTokenTextSplitter,
)

def main():
  # Load environment variables from .env file
  load_dotenv()

  mistral_api_key = os.getenv("MISTRAL_API_KEY")
  if mistral_api_key is None:
    raise ValueError("MISTRAL_API_KEY environment variable is not set.")

  client = Mistral(api_key=mistral_api_key)

  reader = PdfReader("data/NVIDIA-2025-Annual-Report.pdf")
  pdf_texts = [p.extract_text().strip() for p in reader.pages]

  # Filter the empty strings
  pdf_texts = [text for text in pdf_texts if text]

  # split the text into smaller chunks
  character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
  )
  character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

  token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
  )
  token_split_texts = []
  for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
  print(f"Split documents into {len(token_split_texts)} chunks")

  embedding_function = SentenceTransformerEmbeddingFunction()

  chroma_client = chromadb.Client()
  chroma_collection = chroma_client.create_collection(
    "annual-report-collection", embedding_function=embedding_function # type: ignore
  )

  # extract the embeddings of the token_split_texts
  ids = [str(i) for i in range(len(token_split_texts))]
  chroma_collection.add(ids=ids, documents=token_split_texts)
  chroma_collection.count()

  query = "What was the total revenue for the year?"

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

    response = client.chat.complete(
      model=model,
      messages=messages,
    )
    content = response.choices[0].message.content
    return content
  
  original_query = "What was the total profit for the year, and how does it compare to the previous year?"
  hypothetical_answer = augment_query_generated(original_query)

  # Combine the original query with the hypothetical answer
  joint_query = f"{original_query} {hypothetical_answer}"

  results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
  )
  retrieved_documents = results["documents"][0] # type: ignore

	# Function to generate a response from Mistral
  def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
      "You are an assistant for question-answering tasks. Use the following pieces of "
      "retrieved context to answer the question. If you don't know the answer, say that you "
      "don't know. Be detailed in your answer. "
      "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

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
  
  answer = generate_response(original_query, retrieved_documents)
  print("==== Question ====")
  print(f"Oringal Query: {original_query}")
  print("==== Hypothetical Answer ====")
  print(hypothetical_answer)
  print("==== Actual Answer ====")
  print("#######################")
  print(answer)
  
  embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
  umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
  projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

  retrieved_embeddings = results["embeddings"][0] # type: ignore
  original_query_embedding = embedding_function([original_query])
  augmented_query_embedding = embedding_function([joint_query])

  projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
  )
  projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
  )
  projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
  )

  # Plot the projected query and retrieved documents in the embedding space
  plt.figure()

  plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
  )
  plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
  )
  plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
  )
  plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
  )

  plt.gca().set_aspect("equal", "datalim")
  plt.title(f"{original_query}")
  plt.axis("off")
  plt.show()  # display the plot

if __name__ == "__main__":
  main()
