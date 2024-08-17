from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import HumanMessage, AIMessage, BaseMessage

from langchain_community.chat_message_histories import SQLChatMessageHistory


import getpass
import os, sys
import environ
import bs4
import pdb

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder


env = environ.Env()
environ.Env.read_env()

os.environ["OPENAI_API_KEY"] = env("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = env("LANGCHAIN_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

from gMail import gMailReader


"""
The following makes a request to Google Calendar API with the OAuth token
and retrieves number_of_results from the specified date.
"""

from gCalendar import GoogleCalendarReader
from datetime import date

loader = GoogleCalendarReader()
documents = loader.load_data(start_date=date.today(), number_of_results=10)


from typing import List
from langchain.docstore.document import Document as LCDocument

formatted_documents: List[LCDocument] = [doc.to_langchain_format() for doc in documents]


"""
Sets up agent 
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(formatted_documents)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

from langchain.tools.retriever import create_retriever_tool


### Contextualize question ###
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )
# # helpful and friendly assistant for question-answering tasks for WikiWard - a spoiler free Wikipedia site.
# ### Answer question ###
# qa_system_prompt = """\
# Use the following pieces of retrieved context to answer the question. \
# If the retrieved context does not answer the question, just say you don't know. \
# Use three sentences maximum and keep the answer concise.\
# Context: {context}\n\nQuestion: {input}
# """

# # print(qa_system_prompt)
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# this is where llm uses its model to answer the question
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # https://python.langchain.com/v0.1/docs/integrations/memory/sqlite/
# chain_with_history = RunnableWithMessageHistory(
#     rag_chain,
#     # only use chat history from questions asked on the respective page
#     lambda session_id: SQLChatMessageHistory(
#         session_id=1,
#         connection="sqlite:///sqlite.db",
#     ),
#     history_messages_key="chat_history",
#     input_messages_key="input",
#     output_messages_key="answer",
# )
# # This is where we configure the session id
# config = {"configurable": {"session_id": 1}}

# query = "Create a summary for what am I doing on the day: 2024-08-29"


# response = chain_with_history.invoke({"input": query}, config=config)["answer"]

# print(response)

from langchain.tools import Tool


# tool = create_retriever_tool(
#     retriever,
#     "search_calendar_events",
#     "Searches user made events in users calendar",
# )
# tools = [tool, gMailReader.gMailTools()]

tools = gMailReader.gMailTools()

# Define your tools dictionary
# tools = {
#     "context": search_calendar_events_tool,
#     "Gmail_tools": gMailReader.gMailTools(),  # Assuming this is a callable
# }

"""Creates Agent"""

from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)

from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages

# agent = create_openai_tools_agent(llm, tools, prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     # verbose=True,
#     # handle_parsing_errors=True,
#     # memory=chain_with_history,
#     max_iterations=10,
# )
# print(formatted_documents)

# Create a single string that concatenates all the page_content values
all_page_content = " ".join(document.page_content for document in formatted_documents)

example_query = (
    "send an Email to aysmith17@gmail.com of my next event given my schedule: "
    + all_page_content
)
# "Reply to the most recent email from aysmith17@gmail.com"
# "draft to fake@fake.com a good morning email"

# events = agent_executor.invoke(
#     {
#         "input": example_query,
#         # "chat_history": chain_with_history,
#     }
# )
events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
# print(events)
