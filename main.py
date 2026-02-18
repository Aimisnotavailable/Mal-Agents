from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
Be a helpful assistant for coding questions. Use the following reviews to answer the question. If you don't know the answer, say you don't know.

The question is {question}

Be nice and concise in your answer.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    #reviews = retriever.invoke(question)
    result = chain.invoke({"question": question}) # chain.invoke({"reviews": reviews, "question": question})
    print(result)