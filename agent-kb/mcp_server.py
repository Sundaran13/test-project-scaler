from mcp.server.mcpserver import MCPServer
from knowledge_base.vector_store import search

mcp = MCPServer("KnowledgeBaseServer")


@mcp.tool()
def retrieve_knowledge_base(query: str) -> str:
    """
    Searches the knowledge base for information relevant to the query.
    Returns the most relevant document chunks with their sources.
    """
    results = search(query, k=4)

    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] (source: {source})\n{doc.page_content}")

    return "\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run(transport="stdio")