This is a repository where I track my experimental journey with the Retrieval-Augmented Generation (RAG) implemented using LangChain and integrated with a chat interface using Streamlit. 

## Overview

This project is an ongoing experiment exploring the capabilities and applications of combining LangChain's RAG for generating responses with a user-friendly chat interface built with Streamlit. 
The primary goal is to understand, experiment, and possibly identify some improvements over the existing conventions for implementing RAG. 

## Installation

To get started with these experiments on your local setup, please follow the steps below:

```bash
# Clone the repository
git clone [https://github.com/abhijitpal1247/RAG-exp.git](https://github.com/abhijitpal1247/RAG-exp.git)

# Navigate into the repository
cd RAG-exp

# Install the necessary dependencies
pip install -r requirements.txt

Create a .env files with the following environment variables stored in it
WCS_DEMO_RO_KEY
WCS_DEMO_URL
HUGGINGFACEHUB_API_TOKEN
```

I have used weaviate vector database for my experiments. It provides a free demo instance which came in very very handy while conducting these experiments.


## Running the Streamlit UI

After installing the dependencies, you can run the Streamlit UI to interact with the RAG-powered chat system using the following command:

```bash
streamlit run main.py
```
