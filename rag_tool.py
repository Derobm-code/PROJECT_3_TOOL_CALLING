from pathlib import Path

from embeddings import get_embedding
from similarity import cosine_similarity
from chunking import chunk_text

###############################retrieval logic##################################

def load_documents(directory: str) -> dict[str, str]:
    documents = {}

    for path in Path(directory).glob("*.txt"):
        documents[path.name] = path.read_text(
            encoding="utf-8"
        )

    return documents

def create_index(documents: dict[str, str]) -> list[dict]:
    index = []

    for filename, text in documents.items():

        chunks = chunk_text(
            text, chunk_size =100, overlap= 20,
        )

        for chunk_number, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            index.append(
                {
                    "filename": filename,
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "embedding": embedding,
                }
            )

    return index

def search_index(
        query: str,
        index: list[dict],
        top_k: int = 3,
) -> list[dict]:

    query_embedding = get_embedding(query)

    results = []

    for item in index:
        score = cosine_similarity(
            query_embedding,
            item["embedding"],
        )

        results.append(
            {
                "filename": item["filename"],
                "chunk_number": item["chunk_number"],
                "text": item["text"],
                "score": score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

RAG_INDEX = None

def get_rag_index() -> list[dict]:
    global RAG_INDEX

    if RAG_INDEX is None:
        documents = load_documents(str(DOCUMENTS_DIR))

        print(f"Loaded {len(documents)} documents.")

        print("Creating RAG index...")
        RAG_INDEX = create_index(documents)

        print(f"Created {len(RAG_INDEX)} searchable chunks.")
        print("RAG index ready.")

def search_knowledge_base(query: str) -> str:
    """
    Search the knowldge base for information relevant to a query.
    
    Args:
        query: The user's question or search request.
        
    Returns:
        The most relevant document chunks with similarity scores.
    """
    index = get_rag_index()

    results = search_index(
        query, RAG_INDEX, top_k=3,
    )

    if not results:
        return "No relevant information was found."

    output =[]

    for result in results:

        output.append(
            f"Source: {result['filename']}\n"
            f"Chunk: {result['chunk_number']}\n"
            f"Similarity: {result['score']:.4f}\n"
            f"Content: {result['text']}"
        )

    return "n\n".join(output)