from dotenv import load_dotenv
load_dotenv(override=True)

import faiss
from openai import AzureOpenAI
from src.services.models.embeddings import Embeddings
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.ingestion.ingest_files import ingest_files_data_folder
from src.services.models.llm import LLM
import os
from dotenv import load_dotenv
import time

def rag_chatbot(llm: LLM, input_text: str, history: list, index: FAISSIndex):
    """
    Retrieves relevant information from the FAISS index, generates a response using the LLM, and manages the conversation history.
    """
    # 1. Retrieve context from FAISS Index
    try:
        retrieved_chunks = index.retrieve_chunks(query=input_text, num_chunks=5)
        context = "\n".join(retrieved_chunks)
    except Exception as e:
        context = ""
        print(f"Error retrieving context from FAISS: {e}")

    # 2. Pass retrieve context to the LLM along with history
    ai_response = llm.get_response(history, context, input_text)

    # 3. Update conversation history
    history.append({"role": "user", "content": input_text})
    history.append({"role": "assistant", "content": ai_response})

    return ai_response, history


def main():
    """
    Main function to run the chatbot.
    """
    # Initialize embeddings and FAISS index
    embeddings = Embeddings()
    index = FAISSIndex(embeddings=embeddings.get_embeddings)

    # Load or create FAISS index
    try:
        index.load_index()
    except FileNotFoundError:
        print("Index not found. Ingesting documents...")
        ingest_files_data_folder(index)
        index.save_index()

    # Initialize LLM and history
    llm = LLM()
    history = []
    print("\n# INITIALIZED CHATBOT #")

    # Main chat loop
    while True:
        user_input = str(input("You:  "))
        if user_input.lower() == "exit":
            break
        response, history = rag_chatbot(llm, user_input, history, index)
        print("AI: ", response)


if __name__ == "__main__":
    main()



def main():
    """Main function to run the chatbot."""

    embeddings = Embeddings()
    
    index = FAISSIndex(embeddings=embeddings.get_embeddings)

    try:
        index.load_index()
    except FileNotFoundError:
        print("Nao encontrei o ficheiro")
        raise ValueError("Index not found. You must ingest documents first.")


    llm = LLM()
    history = []
    print("\n# INTIALIZED CHATBOT #")

    while True:
        user_input = str(input("You:  "))
        if user_input.lower() == "exit":
            break
        response, history = rag_chatbot(llm, user_input, history, index)
        print("inside while true")
        print("AI: ", response)


if __name__ == "__main__":
    main()