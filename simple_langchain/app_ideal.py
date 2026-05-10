from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def ideal_demo():
    llm = ChatOllama(model="llama3:latest")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise Python tutor."),
        ("human", "Explain what a Python virtual environment is in 2 sentences.")
    ])

    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({}))
