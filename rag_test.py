import os
import glob
import numpy as np
import pandas as pd
import time

# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.schema import Document
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch.nn.functional as F
from torch import Tensor
import torch

print('TORCH GPU:', torch.cuda.is_available())

from transformers import AutoTokenizer, AutoModel

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

#parser = JsonOutputParser(pydantic_object=grade)

prompt = PromptTemplate(
    template=""" Ты эксперт косметолок \n\n.
    Отвечай только по русски.\n\n 
    Избегай повторов. \n\n 
    Отвечай только на вопросы темы косметология. \n\n
    Отвечай на вопросы с помощью полученного контекста:  \n\n {context} \n\n.\n\n 
    Добавь к ответу 3-5 предложений по теме вопроса: {question}.\n\n
    Верни результат в форме JSON, где в поле \'recomended product\' укажи имя имя продукта из контекста, а в поле \'answer\' запиши ответ на вопрос \n\n
    Объясни свои рекомндации: 
    """,
    input_variables=["question","context"])

prompt_list = PromptTemplate(
        template="""Ты эксперт косметолог. \n
        Отвечай только по русски.\n
        Избегай повторов. \n
        Отвечай только на вопросы темы косметология. \n
        Если вопрос не по теме косметология, тогда сообщи, что не можешь ответить
        Используй для рекомендаций только продукты из  контекста:  \n{context} \n
        Добавь к ответу 3-5 предложений по теме вопроса. \n
        Верни результат без какой-либо преамбулы в форме JSON, где в поле \'recomended product\' укажи имена 2-3 продуктов из контекста подходящие {question}, а в поле \'ans>
 
        Это наш вопрос:
        \n ------- \n
        {question}.
        \n ------- \n
         : 
        """,
        input_variables=["question"],
    )

llm = ChatOllama(model='deepseek-r1:32b', temperature=0)
chain = prompt | llm | StrOutputParser() #parser

query = 'Опиши утренний уход за лицом.'
context = ''
better_question = chain.invoke({"question": query, 'context': context})
print(better_question)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embedding(text: str):
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    # Tokenize the input texts
    batch_dict = tokenizere(text, max_length=1024, padding=True, truncation=True, return_tensors='pt')
    
    outputs = modele(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    

    # embedding = embedding_model.encode(text)
    embedding = F.normalize(embeddings, p=2, dim=1)

    return embedding[0,:].detach().numpy()
import numpy as np

def vector_search(user_query,dataset_df):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    collection = dataset_df.embedding.values 
    # print(collection[0][0][:10])
    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    collection_i = []
    ind_drop = []
    for i in range(len(collection)):
        print(len(collection[i]))
        # print(collection[i][0])
        if len(collection[i])>0:
            collection_i.append(collection[i])
        else:    
            ind_drop += [i] 
    if len(ind_drop)>0:
        dataset_df.drop(ind_drop, axis=0, inplace=True)
        print(dataset_df.shape)
    pipeline =   np.vstack( dataset_df.embedding.values) @ np.array(get_embedding(user_query)).reshape(-1) 
    print(pipeline.shape)
    ind = np.argsort(pipeline)
    print(ind[:5])

    # Execute the search
    results = dataset_df.iloc[ind[-4:],:].loc[:, ["text"]].values.tolist()
    print(results)
    return results, ind[-4:]

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # LLM
   

    results_doc = vector_search(question,dataset_df)
    # Data model
    
    # Set up a parser + inject instructions into the prompt template.
    

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for ind, data in list(results_doc.items()):
        filtered_docs.append(json.dumps(data, ensure_ascii=False,))
        
    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "local": local,
            "run_web_search": search,
        }
    }

def transform_query(state, local_llm = llm):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    local = state_dict["local"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""Ты эксперт косметолог. \n
        Отвечай только по русски.\n
        Избегай повторов. \n
        Отвечай только на вопросы темы косметология. \n
        Если вопрос не по теме косметология, тогда сообщи, что не можешь ответить
        Отвечай на вопросы по косметологии с помощью полученного контекста:  \n{context} \n
        Добавь к ответу 3-5 предложений по теме вопроса. \n
        Верни результат без какой-либо преамбулы в форме JSON, где в поле \'recomended product\' укажи имена 2-3 продуктов из контекста подходящие {question}, а в поле \'answer\' запиши ответ на вопрос \n\n
 
        Это наш вопрос:
        \n ------- \n
        {question}.
        \n ------- \n
         : 
        """,
        input_variables=["question"],
    )

    # Grader
    llm = local_llm
    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question, 'context':context})

    return {
        "keys": {"documents": documents, "question": better_question, 
        "local": local}
    }

wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="d", max_results=20, backend='auto')

tool =  DuckDuckGoSearchResults(api_wrapper=wrapper, source='text', output_format="list")

def web_search(question=''): 
    

    docs = tool.invoke({"query": question})
    web_results = docs#"\n".join([d for d in docs])
    web_results = Document(page_content=web_results)
    return web_results


## work
input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

tokenizere = AutoTokenizer.from_pretrained("thenlper/gte-large")
modele = AutoModel.from_pretrained("thenlper/gte-large")

# Tokenize the input texts
batch_dict = tokenizere(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = modele(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# (Optionally) normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())

dataset_df = pd.read_csv('data_goldenaple_prepeared_to_embeding.csv', index_col=0)
print(dataset_df.head())

dataset_df = dataset_df.iloc[:1000, :]

dataset_df["embedding"] = dataset_df["text"].apply(get_embedding)
print(dataset_df.shape)


documents = []


documents, ind = vector_search(query, dataset_df)

web_results = web_search(question=query)
documents.append(web_results)

print(len(documents))

df_list = pd.read_json('titles.json')
documents_list = df_list.iloc[:,0].tolist()

n3 = 3
for model_name in  ['deepseek-r1:32b', 'qwen:32b', 'llama3.3:latest', 'phi4:latest', 'mistral:latest', 'gemma2:27b', 'llama3.2-vision:latest','deepseek-r1:70b', 'deepseek-r1:latest' ]:
    count=0
    print(model_name)
    with open(model_name + '_rez.txt','a+') as f:
        f.write('id,query,  no_context,t0,  context, t1, list_context, t2, docs, web \n')
    for  query in ['Опиши утренний уход за лицом.', 'Опиши методы RAG для анализа документов', "Дай рекомендации по лечению акне",]:
        print(query)
        try:
            llm = ChatOllama(model=model_name, temperature=0.1)
            chain = prompt | llm | StrOutputParser() #parser
            t0 = time.time()
            better_question_no_context = []
            for j in range(n3):
                context = ''
                better_question_no_context.append( chain.invoke({"question": query, 'context': context}))
            t0 = time.time() -  t0
            t0 = t0/n3
            #print(better_question_no_context)

            llm = ChatOllama(model=model_name, temperature=0.1)
            chain = prompt | llm | StrOutputParser() 

            t1 = time.time()

            context = ' ; '.join(documents[0])
            better_question = []
            for j in range(n3):
                better_question.append( chain.invoke({"question": query, 'context': context}))
            t1 = (time.time() -  t1)/n3
            #print(better_question)

            llm = ChatOllama(model=model_name, temperature=0.1)
            chain = prompt_list | llm | StrOutputParser() 
            better_question_list = []
            t2 = time.time()
            for j in range(n3):
                context = ' ; '.join(documents_list)
                better_question_list.append(chain.invoke({"question": query, 'context': context}))
            t2 = (time.time() -  t2)/n3
            #print(better_question_list)
            with open(model_name + '_rez.txt', 'a+') as f:
                for j in range(n3):
                    s = ', '.join([str(count), query, better_question_no_context[j], str(t0), better_question[j], str(t1), better_question_list[j], str(t2), '['+','.join([str(i) for i in ind]) + ']', web_results.page_content]) 
                    f.write(s + ' \n')
                    count += 1
        except:
            print(model_name + ': error')
