import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
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
def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(transcript)
    return chunks

# 3️⃣ Embeddings & Vector Store
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# 4️⃣ RAG QnA with Groq LLaMA 3 (retrieval + augmentation + generation)
def rag_qa(query, vectorstore, groq_client):
    print("\n🔍 Retrieving relevant chunks for your query...\n")
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        print("❌ No relevant chunks found for your query.")
        return

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_rag_prompt(context, query)

    print("\n📝 Sending prompt to LLaMA 3 via Groq...\n")
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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

        chunks = split_transcript(transcript)
        print(f"✅ Transcript split into {len(chunks)} chunks.")

        vectorstore = embed_and_store(chunks)
        print("✅ Embeddings created and stored in FAISS vectorstore.")

    else:
        print("❌ Transcript fetch failed, exiting.")
        exit()

    while True:
        user_question = input("\n❓ Ask a question about the video (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        rag_qa(user_question, vectorstore, groq_client)
