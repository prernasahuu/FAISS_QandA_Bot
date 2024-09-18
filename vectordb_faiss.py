# -*- coding: utf-8 -*-
"""VectorDB-faiss.ipynb
by Colab.
Original file is located at
    https://colab.research.google.com/drive/1qDZVgvupxgVEU3qluzXLN12w_WinTeca
"""

#installing packages
!pip install requirements.txt

para= """input"""
#splitting the paragraph via split (.)
sentences= para.split(',')
print(sentences)

#import libaries
import os
import faiss
from sentence_transformers import SentenceTransformer

#Hugging-Face-Token API
from google.colab import userdata
HF_TOKEN=userdata.get('HF_TOKEN')

#initialising the miniLM version 6 word embedding model.(used hugging-face API)
model = SentenceTransformer('msmarco-MiniLM-L-6-v3')

#encoding the tokens/chunks with model.
embeddings= model.encode(sentences)
print(embeddings)

embeddings.shape

dimension = embeddings.shape[1] #as we are taking 1 sentence as a charateristic/deminsion.
print(dimension)
index = faiss.IndexFlatL2(dimension) #euclidean distance to vectorize the sentences according to similarities.
index.add(embeddings) #adding the embeddings into the faiss DB

"""#Direct approach to get efficient and accurate retrieval of answers based on the query asked.
"""
k=3 #top 3 results with respect to the query asked.
distances, indices = index.search(embeddings, k)

query= input("Ask the question: ")
query_emb= model.encode(query)
for i in range(k):
  print(f"{i+1}.{sentences[indices[0][i]]}(Distance: {distances[0][i]:.4f}")

"""#Retrival of the answer to the query, using gpt 3.5 turbo API.
##This provides an domain specific answer to the query.
"""
#GPT 3.5 turbo API
from google.colab import userdata
API_KEY=userdata.get('API_KEY_MY')

import openai
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = API_KEY
def generate_response(prompt):
    client = OpenAI()
    chat_completion = client.chat.completions.create(
    messages=[{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": prompt}],
    model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

prefix="Answer the query specific to the given context."

prompt=f'''{prefix}\n\n\
Question: {query}\n\
Context: {sentences}\n\
Answer:'''

#GPT answer to the query.
print(generate_response(prompt))

"""#Interfacing Gradio with GPT 3.5 turbo."""

import gradio as gr
def chat_with_bot(query):
    return generate_response(query)

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="Enter the Query"),
    outputs=gr.Textbox(label="Response"),
    title="Q&A Bot",
    description="Ask anything and get response specific to the context,only."
)

# Launch the interface
iface.launch(shape=True)

