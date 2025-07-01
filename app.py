import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os
import time
import pickle

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Create folder for saved embeddings
os.makedirs("saved_embeddings", exist_ok=True)

# Helper functions
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([item["text"] for item in transcript])

def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(transcript)

def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

def build_rag_prompt(context, question):
    return f"""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""

def rag_qa(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_rag_prompt(context, query)
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response

# Streamlit page setup
st.set_page_config(page_title="🎬 YouTube RAG Chatbot", page_icon="🤖")
st.title("🎬 YouTube RAG Chatbot 🤖")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Step 1: Get YouTube Video URL/ID
if st.session_state.vectorstore is None:
    st.subheader("Step 1: Paste YouTube Video URL or ID")
    video_input = st.text_input("Paste here", key="video_input")

    if st.button("Fetch Transcript and Prepare Embeddings"):
        if video_input:
            try:
                video_id = video_input.split("v=")[-1] if "watch?v=" in video_input else video_input
                embed_path = f"saved_embeddings/{video_id}.pkl"

                if os.path.exists(embed_path):
                    with open(embed_path, "rb") as f:
                        st.session_state.vectorstore = pickle.load(f)
                    st.success("✅ Loaded saved embeddings for this video.")
                else:
                    with st.spinner("Fetching transcript and preparing embeddings..."):
                        transcript = get_transcript(video_id)
                        chunks = split_transcript(transcript)
                        vectorstore = embed_and_store(chunks)

                        # Save for session memory
                        with open(embed_path, "wb") as f:
                            pickle.dump(vectorstore, f)

                        st.session_state.vectorstore = vectorstore

                    st.success("✅ Video processed and embeddings saved! Now you can chat below.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please paste a YouTube URL or video ID before clicking the button.")

# Download embeddings button
if st.session_state.vectorstore is not None:
    st.download_button(
        label="📥 Download Embeddings (.pkl)",
        data=pickle.dumps(st.session_state.vectorstore),
        file_name="youtube_embeddings.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )

# Step 2: Chat interface
if st.session_state.vectorstore is not None:
    st.subheader("Step 2: Ask questions about the video below")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Type your question and press Enter:")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            with st.spinner("Thinking..."):
                try:
                    response = rag_qa(user_question, st.session_state.vectorstore)
                    for chunk in response:
                        delta = chunk.choices[0].delta.content or ""
                        streamed_text += delta
                        placeholder.markdown(streamed_text + "▌")
                        time.sleep(0.01)
                    placeholder.markdown(streamed_text)
                except Exception as e:
                    placeholder.error(f"❌ Error: {e}")
                    streamed_text = f"❌ Error: {e}"

            st.session_state.messages.append({"role": "assistant", "content": streamed_text})

# Reset button
st.divider()
if st.button("🗑️ Clear Chat & Reset", use_container_width=True, type="secondary"):
    st.session_state.clear()
    st.rerun()
