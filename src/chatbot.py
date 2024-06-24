import gradio as gr
from model_utils import *
from dotenv import load_dotenv

import os, gc, shutil
import gradio as gr
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch
from model_setup import ModelSetup
import time
from threading import Thread


def load_models(embedding_model, llm, do_quantize):
    global qa
    global streamer
    model_setup = ModelSetup(embedding_model, llm, do_quantize)
    success_prompt = model_setup.setup()
    qa, streamer = conv_qa.create_conversation(
        model_setup.model,
        model_setup.tokenizer,
        model_setup.vectordb,
    )
    return success_prompt

def get_chat_history(inputs):
    res = []
    for human, ai in inputs:
        res.append(f"user: {human}\nassistant: {ai}\n")
    return "\n".join(res)


def add_text(history, text):
    history = history + [[text, None]]
    return history, ""


conv_qa = Conversation_RAG()


def bot(
    history,
    instruction="Bạn là trợ lý ảo thông minh. Bạn sẽ đọc nội dung từ ngữ cảnh để trả lời câu hỏi của người dùng. Trả lời ngắn gọn, đủ ý, nội dung bằng tiếng Việt.",
    temperature=0.1,
    max_new_tokens=1024,
    repetition_penalty=1.1,
    top_k=10,
    top_p=0.95,
    k_context=5,
    num_return_sequences=1,
):
    chat_history_formatted = get_chat_history(history[:-1])

    def run_qa():
        qa.invoke({"question": history[-1][0], "chat_history": ""})

    t = Thread(target=run_qa)
    t.start()

    history[-1][1] = ""
    for new_text in streamer:
        history[-1][1] += new_text
        yield history

    t.join()

    # res = qa.invoke(
    #     {"question": history[-1][0], "chat_history": chat_history_formatted}
    # )


def reset_sys_instruction(instruction):
    default_inst = "Bạn là trợ lý ảo thông minh. Bạn sẽ đọc nội dung từ ngữ cảnh để trả lời câu hỏi của người dùng. Trả lời ngắn gọn, đủ ý, nội dung bằng tiếng Việt."
    return default_inst


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return None


with gr.Blocks(
    gr.themes.Soft(
        primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple
    )
) as demo:
    gr.Markdown(
        """# Retrieval Augmented Generation - Chat for Vietnamese \n
                Kết hợp RAG với VinaLLama trong việc ứng dụng chatbot vào hệ thống giáo dục.
                """
    )
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Chatbox",
                height=500,
            )

            txt = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Enter your question and press shift+enter ",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit", variant="primary", size="sm")

                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear", variant="stop", size="sm")

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot], chatbot
            )
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot,
                [
                    chatbot,
                ],
                chatbot,
            ).then(clear_cuda_cache, None, None)

            clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    load_dotenv()
    print("Load model...")
    load_outcome = load_models(
        embedding_model=os.environ['EMBEDDING_MODEL'],
        llm=os.environ['LLM'],
        do_quantize=False
    )
    print(load_outcome)

    demo.queue()
    demo.launch(max_threads=3, debug=True, share=True)

