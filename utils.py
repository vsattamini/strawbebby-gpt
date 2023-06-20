import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import tiktoken

CHAT_API_PARAMS = {
    "temperature": 0,
    "stop": ["Client:", "Agent:"]
}

REFLECTION_API_PARAMS = {
    "temperature": 0,
    "model": 'gpt-3.5-turbo-16k',
    "stop": ["Client:", "Agent:"]
}
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")# The encoding scheme to use for tokenization


def get_embedding(text, model="text-embedding-ada-002"):
    embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    return embedding

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, embeddings_pkl):
    query_embedding = get_embedding(query, "text-embedding-ada-002")

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in embeddings_pkl.items()
        if vector_similarity(query_embedding, doc_embedding) > .5
    ], reverse=True)

    return document_similarities

def get_context(user_message,embeddings_pkl,df,context_amount=20):
    document_similarities = order_document_sections_by_query_similarity(user_message,embeddings_pkl)
    similarities = [document_similarities[i][0] for i in range(context_amount)]
    cutoff = np.mean(similarities)
    indexes = [document_similarities[i][1] for i in range(context_amount) if similarities[i]>=cutoff]
    context = [f"{df['chunk'].loc[i]}\n(source: {df['document_name'].loc[i]}\n\n)" for i in indexes]   
    return context

def count_tokens(text):
    count = len(tokenizer.encode(text))
    return count

def run_prompt(prompt,params=REFLECTION_API_PARAMS):
    response = openai.ChatCompletion.create(
              **params,
              messages=[
                {"role": "system", "content": prompt}                ]
               )
    answer = response['choices'][0]['message']['content']
    print(response['usage']['total_tokens'])
    return answer

def past_reflect(message_history):
    prompt = f"""You aare an assistant researcher having a conversation with the main reasearching neuro-otologist.
You look back and think about everything that has aleady been talked about and write a summary for yourself.
Take account of everything that has been said and extract the most important bits of information, as well as a summary of what was asked and answered.
Keep it to 3 paragraphs

Message History:
{message_history}

Summary:"""
    answer = run_prompt(prompt)
    return(answer)
    