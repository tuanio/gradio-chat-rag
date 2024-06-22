import os, gc, shutil
import gradio as gr
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch
from model_setup import ModelSetup
import time
from threading import Thread


def load_models(embedding_model, llm, do_quantize):
    global model_setup
    model_setup = ModelSetup(embedding_model, llm, do_quantize)
    success_prompt = model_setup.setup()
    return success_prompt


def upload_and_create_vector_store(file, embedding_model):
    # Save the uploaded file to a permanent location
    file_path = file.name
    split_file_name = file_path.split("/")
    file_name = split_file_name[-1]

    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.dirname(current_folder)
    data_folder = os.path.join(root_folder, "data")
    permanent_file_path = os.path.join(data_folder, file_name)
    shutil.copy(file.name, permanent_file_path)

    # Access the path of the saved file
    print(f"File saved to: {permanent_file_path}")

    index_success_msg = create_vector_store_index(
        permanent_file_path, embedding_model
    )
    return index_success_msg


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

    qa, streamer = conv_qa.create_conversation(
        model_setup.model,
        model_setup.tokenizer,
        model_setup.vectordb,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        k_context=k_context,
        num_return_sequences=num_return_sequences,
        instruction=instruction,
    )

    chat_history_formatted = get_chat_history(history[:-1])

    def run_qa():
        qa.invoke({"question": history[-1][0], "chat_history": ''})

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

