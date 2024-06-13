import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")


from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from typing import List

from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# https://smith.langchain.com/hub/efriis/self-rag-retrieval-grader
grade_prompt = hub.pull("efriis/self-rag-retrieval-grader")

# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

retrieval_grader = grade_prompt | structured_llm_grader


### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def basic_rag(prompt: str, chat_history: List[dict]):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Combine the rewritten question with the history
    combined_question = f"Chat History:\n{history_str}\nHere is the initial question: \n\n {prompt}"

    # Add to vectorDB
    vectorstore = PineconeVectorStore(
        embedding=OpenAIEmbeddings(),
        index_name="airbus",
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(prompt)

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    response = ""
    # Run
    for update in rag_chain.stream({"context": docs, "question": combined_question}):
        response += update
        yield {"full_response": response}
