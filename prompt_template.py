def build_rag_prompt(context, question):
    """
    Builds a clean, structured prompt for RAG generation using Groq LLaMA 3.
    """
    prompt = f"""
You are a precise and helpful assistant. Answer accurately using ONLY the provided context.
If the context does not contain the answer, reply: "I don't know."

Provide a structured answer:
- Direct and clear.
- Bullet points if applicable.
- Reference examples from the context where possible.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
"""
    return prompt
