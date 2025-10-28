from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from typing import Annotated, TypedDict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Carregar vector storage
def get_embed():
    if not hasattr(get_embed,'_instance'):
        get_embed._instance = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-mpnet-base-v2')
    return get_embed._instance

emb_function = get_embed()

vector_store = FAISS.load_local('vstore_faiss', emb_function, allow_dangerous_deserialization= True)
instructions_rag = open(r'prompt_rag.txt').read().replace('\n','').strip()

def similarity_search(query: str, k: int = 3) -> str:
    """Busca semântica na documentação técnica de investimentos."""
    results = vector_store.similarity_search(query, k=k)
    contents = [x.page_content for x in results]
    return "\n\n".join(contents)

tool_similarity_search = StructuredTool.from_function(
    func=similarity_search,
    name="similarity_search",
    description="Busca semântica na documentação técnica de investimentos."
)

class MessagesStateRag(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    model_decision: str
    model : object

def generate_response(state:MessagesStateRag):
    'Gera resposta baseado no estado RAG do agente deverá usar a tool similarity search ou apenas responder'
    model = state['model']
    messages = [SystemMessage(content=instructions_rag)] + state['messages']
    response = model.bind_tools([tool_similarity_search]).invoke(messages)
    return {'messages': [response]}

def graph_builder_rag(state:MessagesStateRag):
    graph_builder = StateGraph(MessagesStateRag)
    graph_builder.add_node('agent_rag', generate_response)
    graph_builder.add_node('tools', ToolNode(tools = [similarity_search]))

    graph_builder.add_conditional_edges('agent_rag', tools_condition,'tools')
    graph_builder.add_edge('tools','agent_rag')
    graph_builder.add_edge(START,'agent_rag')

    graph_rag = graph_builder.compile()

    return graph_rag

# graph_builder_rag = graph_builder_rag()
# display(Image(graph.get_graph().draw_png()))

# display(Image(graph_rag.get_graph().draw_mermaid_png()))
