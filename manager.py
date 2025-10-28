from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from langchain_core.messages import AIMessage,  SystemMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun

from IPython.display import Image, display
from agente_rag import graph_builder_rag

from dotenv import load_dotenv
from typing import Annotated, TypedDict
import logging

load_dotenv(dotenv_path='.env')


class MessagesState_manager(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    model_decision: str
    model : object
    
class Manager():
    def __init__(self):
        self.instructions_manager = open(r'prompt_manager.txt').read().replace('\n','').strip()
        self.instructions_validator = open(r'prompt_validator.txt').read().replace('\n','').strip()

    def _normalize_result(self, result):
        """
        Garante que o retorno tenha conteúdo textual serializável.
        - Converte listas/chunks em string Markdown.
        - Substitui None por mensagem padrão.
        - Retorna o dicionário original, mas com content limpo.
        """
        try:
            messages = result.get("messages", [])
            if not messages:
                result["messages"] = [AIMessage(content="⚠️ Nenhuma mensagem retornada pelo modelo.")]
                return result

            msg = messages[-1].content

            # Caso o modelo retorne lista de chunks
            if isinstance(msg, list):
                msg = "\n".join(str(m.get("text", m)) for m in msg if m)

            # Caso o modelo retorne None
            elif msg is None:
                msg = "⚠️ Nenhum conteúdo retornado pelo modelo."

            # Força conversão para string
            messages[-1].content = str(msg)
            result["messages"] = messages
            return result

        except Exception as e:
            logging.exception("Erro ao normalizar conteúdo: %s", e)
            return {"messages": [AIMessage(content=f"Erro ao normalizar conteúdo: {e}")]}   

    def orquestrador(self, state:MessagesState_manager):
        ''' Gera resposta baseado no que for perguntado sobre as tools ou sobre a documentação tecnica'
            retorna
            "rag"
            'end'
            "tool"
        '''
        model = state['model']
        messages = [SystemMessage(content=self.instructions_manager)] + state['messages']
        response = model.invoke(messages)
        result = {'messages': [response]}
        return self._normalize_result(result)
    
    def search_tool_duck(self, state:MessagesState_manager):
        """
        Executa a busca no DuckDuckGo e retorna o conteúdo.
        """
        query = state["messages"][-1].content.replace('tool','')  # última mensagem do usuário
        print(query)
        result = DuckDuckGoSearchRun().run(query)  
        return self._normalize_result({"messages": [AIMessage(content=result)]})

    def validator(self, state:MessagesState_manager):
        """
        Estrutura e valida a resposta final e entrega para o usuário 
        """
        model = state['model']
        messages = [SystemMessage(content=self.instructions_validator)] + state['messages']
        response = model.invoke(messages)
        result = {'messages': [response]}
        return self._normalize_result(result)

    def router_node(self, state:MessagesState_manager):
        last_msg = state['messages'][-1].content.lower()
        if 'rag_flag' in last_msg:
            return "rag"
        elif 'tool_flag' in last_msg:
            return "tool"
        else:
            return "straight"
        
    def graph_builder_manager(self): 
        graph_builder = StateGraph(MessagesState_manager)
        graph_builder.add_node('orquestrador', self.orquestrador)
        graph_builder.add_node('agent_rag', graph_builder_rag)
        graph_builder.add_node('tools', self.search_tool_duck)
        graph_builder.add_node('validator', self.validator)

        graph_builder.add_edge(START,'orquestrador')
        graph_builder.add_conditional_edges('orquestrador', self.router_node,{'rag':'agent_rag',
                                                                        'tool': 'tools',
                                                                        'straight' : END})
        
        graph_builder.add_edge('agent_rag', END)
        graph_builder.add_edge('tools', 'validator')
        graph_builder.add_edge('validator', END)

        memory = MemorySaver()
        graph_manager = graph_builder.compile(checkpointer=memory)

        return graph_manager
    