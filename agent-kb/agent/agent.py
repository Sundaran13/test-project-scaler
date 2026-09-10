import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from agent.tools import retrieve_knowledge_base

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using
a knowledge base. Always use the retrieve_knowledge_base tool to look up
relevant information before answering factual questions. If the knowledge
base doesn't contain relevant information, say so honestly instead of
making something up.

Each retrieved chunk is labelled with its source. When you use information
from a chunk, cite the source inline in parentheses, e.g. (source:
https://www.scaler.com/about/) or (source: scaler_kb.pdf, page 2).
End your answer with a "Sources:" line listing every source you used."""


def build_agent():
    """
    Builds and returns a LangChain agent wired up with:
    - Groq's free LLM API as the reasoning engine
    - the retrieval tool from tools.py
    The agent decides on its own whether/when to call the tool.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=[retrieve_knowledge_base],
        system_prompt=SYSTEM_PROMPT
    )

    return agent