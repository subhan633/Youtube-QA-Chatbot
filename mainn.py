import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain.docstore.document import Document
from prompt_template import build_rag_prompt


# Load environment
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


# 1️⃣ Transcript Extraction
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item['text'] for item in transcript])
        return full_text
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return ""

# 2️⃣ Text Splitting
def split_transcript_with_metadata(transcript_list):
    """
    Splits transcript into chunks with metadata (timestamps for each chunk).
    """
    combined_chunks = []
    current_chunk = ""
    current_start = None

    for entry in transcript_list:
        text = entry["text"].replace("\n", " ")
        start = entry["start"]

        if current_start is None:
            current_start = start

        if len(current_chunk) + len(text) <= 500:
            current_chunk += " " + text
        else:
            combined_chunks.append({
                "content": current_chunk.strip(),
                "metadata": {"start_time": current_start}
            })
            current_chunk = text
            current_start = start

    if current_chunk:
        combined_chunks.append({
            "content": current_chunk.strip(),
            "metadata": {"start_time": current_start}
        })

    return combined_chunks


# 3️⃣ Embeddings & Vector Store
def embed_and_store_with_metadata(chunks_with_metadata):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks_with_metadata
    ]
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    return vectorstore


# 4️⃣ RAG QnA with Groq LLaMA 3 (retrieval + augmentation + generation)
def rag_qa(query, vectorstore, groq_client):
    print("\n🔍 Retrieving relevant chunks for your query...\n")
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        print("❌ No relevant chunks found for your query.")
        return

    # Display retrieved chunks with metadata
    print(f"✅ Retrieved {len(docs)} relevant chunks:\n")
    for idx, doc in enumerate(docs, 1):
        start_time = doc.metadata.get("start_time", "Unknown")
        print(f"--- Chunk {idx} (Start Time: {start_time} sec) ---")
        print(doc.page_content[:500])  # show first 500 chars for clean output
        print("\n")

    # Prepare combined context for LLM
    context = "\n\n".join([
        f"(Start Time: {doc.metadata.get('start_time', 'Unknown')} sec)\n{doc.page_content}"
        for doc in docs
    ])

    # Build structured prompt
    prompt = build_rag_prompt(context, query)

    print("\n📝 Sending prompt to LLaMA 3 via Groq...\n")
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a precise assistant. Use the provided context only to answer."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()
    print(f"\n🤖 Answer:\n{answer}\n")


# MAIN EXECUTION
if __name__ == "__main__":
    # Initialize Groq client
    groq_client = Groq(api_key=groq_api_key)

    # Get transcript from YouTube
    video_id = "mZUG0pr5hBo"  # Change to your desired video
    transcript = get_transcript(video_id)

    if transcript:
        print("✅ Transcript fetched successfully.")

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        chunks_with_metadata = split_transcript_with_metadata(transcript_list)
        print(f"✅ Transcript split into {len(chunks_with_metadata)} chunks with metadata.")

        vectorstore = embed_and_store_with_metadata(chunks_with_metadata)
        print("✅ Embeddings with metadata created and stored in FAISS vectorstore.")

    else:
        print("❌ Transcript fetch failed, exiting.")
        exit()

    while True:
        user_question = input("\n❓ Ask a question about the video (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        rag_qa(user_question, vectorstore, groq_client)
