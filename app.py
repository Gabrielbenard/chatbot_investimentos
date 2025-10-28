import gradio as gr
from dotenv import load_dotenv
import asyncio
load_dotenv(override=True)
from manager import Manager
from langchain_core.messages import HumanMessage
from models import model_gemini, model_qwen
import logging
from typing import Any

logger = logging.getLogger(__name__)

config = {'configurable': {'thread_id': '05'}}
graph_manager = Manager().graph_builder_manager()
selected_model = "both" 

def select_model(choice):
    global selected_model
    selected_model = choice
    return f"✅ Modelo selecionado: {choice.upper()}"

async def run(user_input:str):
    if selected_model == "gemini":
        result = await graph_manager.ainvoke(
            {"messages": [HumanMessage(content=user_input)], "model": model_gemini},
            config=config
        )
        print(type(result))
        print(result)
    elif selected_model == "qwen":
        result = await graph_manager.ainvoke(
            {"messages": [HumanMessage(content=user_input)], "model": model_qwen},
            config=config
        )
    else:  
            # Roda os dois modelos em paralelo
        task1 = asyncio.create_task(
            graph_manager.ainvoke(
                {"messages": [HumanMessage(content=user_input)], "model": model_gemini},
                config=config
            )
        )

        task2 = asyncio.create_task(
            graph_manager.ainvoke(
                {"messages": [HumanMessage(content=user_input)], "model": model_qwen},
                config=config
            )
        )

        result1, result2 = await asyncio.gather(task1, task2)

        resp1 = result1["messages"][-1].content
        resp2 = result2["messages"][-1].content

        # Consolida as respostas
        combined = (
            f"### 🤖 Gemini (Gemini-2.5-Flash)\n{resp1}\n\n"
            f"### 🧠 Qwen (Qwen3-32B)\n{resp2}\n\n"
            f"### 🔍 Análise combinada\n"
            f"- As respostas acima foram geradas por ambos os modelos.\n"
            f"- Utilize a análise cruzada para validar consistência."
        )

        return combined

    return result["messages"][-1].content


# Interface Gradio
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    with gr.Row():
        model_status = gr.Markdown("✅ Modelo selecionado: BOTH")

    with gr.Row():
        gemini_button = gr.Button("Gemini", variant="secondary")
        qwen_button = gr.Button("Qwen", variant="secondary")
        both_button = gr.Button("Ambos", variant="secondary")
         
    gr.Markdown("# Agente investidor")
    query_textbox = gr.Textbox(label="O que gostaria de perguntar?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")

    gemini_button.click(fn=select_model, inputs=gr.Textbox(value="gemini", visible=False), outputs=model_status)
    qwen_button.click(fn=select_model, inputs=gr.Textbox(value="qwen", visible=False), outputs=model_status)
    both_button.click(fn=select_model, inputs=gr.Textbox(value="both", visible=False), outputs=model_status)

    gr.Markdown("---")

    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inline=True)