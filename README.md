# Tymeline Customer Chat Assistant

This project demonstrates the creation of a customer chat assistant for Tymeline using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). The assistant is fine-tuned with specific Tymeline data and implemented in a Streamlit web interface.

## Features
- **Data Collection**: Prepares and processes data related to Tymeline for training the model.
- **LLM Fine-Tuning**: Fine-tunes a pre-trained LLM model (e.g., T5) to align with Tymeline's domain-specific queries.
- **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information using FAISS and OpenAI embeddings.
- **Streamlit Interface**: Provides a simple web-based chat interface for interacting with the assistant.

## Prerequisites
- Python 3.7 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dearnidhi/Tymeline-RAG-Chat.git
   cd Tymeline-RAG-Chat



2. Install the required Python packages:


pip install -r requirements.txt


3. Download the tymeline_data.csv file and place it in the project directory.

4. Run the Streamlit App:

streamlit run app.py

5. Interact with the Chat Assistant: Use the Streamlit interface to ask questions related to Tymeline, and the assistant will provide accurate responses using the fine-tuned LLM and RAG system.

## Project Structure

app.py: Main Python script for the project.
tymeline_data.csv: Data file used for training and retrieval.
requirements.txt: Contains the list of required Python libraries.
README.md: Documentation for the project.


## Technologies Used
Transformers: For handling LLM-related tasks.

Datasets: For managing and preparing data.

FAISS: For efficient vector-based retrieval.

Langchain: For implementing the RAG mechanism.

Streamlit: For building the web interface.

## Future Enhancements
Improve the data collection process by incorporating more diverse customer queries.

Implement more advanced fine-tuning techniques to further enhance the model’s performance.

Add support for multi-turn conversations in the chat interface.
