import os
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")


from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, StateGraph

backend_details = ""

# use pinecone movies database

# Add to vectorDB
vectorstore = PineconeVectorStore(
    embedding=OpenAIEmbeddings(),
    index_name="annualreport",
)
retriever = vectorstore.as_retriever()


### Retrieval Grader

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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

retrieval_grader = grade_prompt | structured_llm_grader


### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# Chain
rag_chain = prompt | llm | StrOutputParser()


### Hallucination Grader

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


### Question Re-writer

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        attempt: current number of attempts to generate
    """

    question: str
    generation: str
    documents: List[str]
    attempt: int = 0


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    global backend_details
    backend_details += "---RETRIEVE---\n\n"
    question = state["question"]
    #backend_details += "attempt: "+str(state["attempt"])+"\n\n"
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    global backend_details
    backend_details += "---GENERATE---\n\n"
    question = state["question"]
    documents = state["documents"]
    attempt = state["attempt"] + 1 

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "attempt": attempt}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    global backend_details
    backend_details += "----CHECK DOCUMENT RELEVANCE TO QUESTION---\n\n"
    question = state["question"]
    documents = state["documents"]
    attempt = state["attempt"] + 1 

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            backend_details += "---GRADE: DOCUMENT RELEVANT---\n\n"
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            backend_details += "---GRADE: DOCUMENT NOT RELEVANT---\n\n"
            continue
    return {"documents": filtered_docs, "question": question, "attempt": attempt}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    global backend_details
    backend_details += "---TRANSFORM QUERY---\n\n"
    question = state["question"]
    documents = state["documents"]
    attempt = state["attempt"] + 1

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print(better_question)
    backend_details += "Better question: "+better_question+"\n\n"
    return {"documents": documents, "question": better_question, "attempt": attempt}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    global backend_details
    backend_details += "---ASSESS GRADED DOCUMENTS---\n\n"
    question = state["question"]
    filtered_documents = state["documents"]
    attempt = state["attempt"]

    backend_details += "attempt: "+str(attempt)+"\n\n"
    if attempt >= 9:
        return "generate"

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        backend_details += "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---\n\n"
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---We have relevant documents, so generate answer---")
        backend_details += "---We have relevant documents, so generate answer---\n\n"
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    global backend_details
    backend_details += "---CHECK HALLUCINATIONS---\n\n"
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempt = state["attempt"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    backend_details += "attempt: "+str(attempt)+"\n\n"
    if attempt >= 7:
        return "exit"

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        backend_details += "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---\n\n"
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        backend_details += "---GRADE GENERATION vs QUESTION---\n\n"
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            backend_details += "---DECISION: GENERATION ADDRESSES QUESTION---\n\n"
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            backend_details += "---DECISION: GENERATION DOES NOT ADDRESS QUESTION---\n\n"
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        backend_details += "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---\n\n"
        
        return "not supported"
    



workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "exit": END,
    },
)

# Compile
app = workflow.compile()


from pprint import pprint


def self_reflective_rag(prompt: str, chat_history: List[dict]) -> Generator[dict, None, None]:
    global backend_details
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    combined_question = f"Chat History:\n{history_str}\nHere is the initial question: \n\n {prompt}"
    backend_details_full = ""
    backend_details = ""
    inputs = {"question": combined_question, "attempt": 0 } 
    for output in app.stream(inputs, {"recursion_limit": 20}):
        for key, value in output.items():
            backend_details += (f"Node '{key}'\n\n")
            yield {"full_response": value.get("generation", ""), "backend_details": backend_details}
            backend_details_full += backend_details
            backend_details = ""
        pprint("\n\n")

    # Final generation
    #full_response = value["generation"]
    #yield {"full_response": full_response, "backend_details": backend_details_full}