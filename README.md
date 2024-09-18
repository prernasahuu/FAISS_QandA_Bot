# Question And Answer_Bot
###Objective: 
To build a Question and Answer Bot that will help to generate response to query on the basis of given Context.
In this project, I have used word embeddings using the MiniLM-v6 model and integrated it with the FAISS vector database to build a robust RAG pipeline. The input for the bot was taken as paragraphs("context"), allowing the model to understand context and provide domain specific and top ranked answers to the query efficiently.

###About Vector-DataBase:
FAISS (Facebook AI Similarity Search) is an open-source library designed for efficient similarity search and clustering of dense vectors. This enabled me to build a responsive and scalable Q&A system that excels at finding relevant information from large amounts of data.
It can handle massive datasets, making it highly scalable and provides fast, memory-efficient algorithms for quick Approximate Nearest Neighbor (ANN) searches. This enabled me to build a responsive and scalable Q&A system that excels at finding relevant information from large amounts of data.

###About Word Embedding Model:
The MiniLMv6 (Mini Language Model version 6) is a compact transformer-based language model designed for efficient natural language processing tasks. It is part of Microsoft's family of models optimized for tasks like text embedding, similarity search, and more.
MiniLMv6 model is used for both the user's query and pre-written context in the database that are transformed into embeddings (vector representations) that capture their semantic meaning. The system compares the query's embedding to the pre-embedded context using cosine/dot product/Euclidean Distance similarity, measuring how close the vectors are to each other in terms of meaning. The answers with the highest similarity scores are ranked and retrieved as the most relevant and spcidfic answer to the query. This approach enables efficient and accurate retrieval of answers based on the meaning of the query, rather than just keyword matching.

###Gradio:
Gradio is used to interface gpt 3.5 turbo responses.(for questions and answers)

###Results:
As it consumes alot of time for students or any person, to go through the whole context, so as to find out an answer to the specific query. Hence, this education based tool,provides relaxation in time comsumption and manual efforts, with the atmost flexibility.
