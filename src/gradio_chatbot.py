import gradio as gr
from model_utils import *
from dotenv import load_dotenv


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
            gr.Markdown("## Upload Document & Select the Embedding Model")
            file = gr.File(type="filepath")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, variant="panel"):
                    embedding_model = gr.Dropdown(
                        choices=[
                            "Fsoft-AIC/videberta-base",
                            "Fsoft-AIC/videberta-xsmall",
                            "vinai/phobert-large",
                            "vinai/phobert-base",
                            "bkai-foundation-models/vietnamese-bi-encoder",
                            "openai-model",
                        ],
                        value="openai-model",
                        label="Select the embedding model",
                    )

                with gr.Column(scale=1, variant="compact"):
                    vector_index_btn = gr.Button(
                        "Create vector store", variant="primary", scale=1
                    )
                    vector_index_msg_out = gr.Textbox(
                        show_label=False,
                        lines=1,
                        scale=1,
                        placeholder="Creating vectore store ...",
                    )

            instruction = gr.Textbox(
                label="System instruction",
                lines=3,
                value="Bạn là trợ lý ảo thông minh. Bạn sẽ đọc nội dung từ ngữ cảnh để trả lời câu hỏi của người dùng. Trả lời ngắn gọn, đủ ý, nội dung bằng tiếng Việt.",
            )
            reset_inst_btn = gr.Button("Reset", variant="primary", size="sm")

            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(
                    label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05
                )
                max_new_tokens = gr.Slider(
                    label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1
                )
                repetition_penalty = gr.Slider(
                    label="repetition_penalty",
                    minimum=0,
                    maximum=2,
                    value=1.1,
                    step=0.1,
                )
                top_k = gr.Slider(
                    label="top_k", minimum=1, maximum=1000, value=10, step=1
                )
                top_p = gr.Slider(
                    label="top_p", minimum=0, maximum=1, value=0.95, step=0.05
                )
                k_context = gr.Slider(
                    label="k_context", minimum=1, maximum=15, value=5, step=1
                )

            vector_index_btn.click(
                upload_and_create_vector_store,
                [file, embedding_model],
                vector_index_msg_out,
            )
            reset_inst_btn.click(reset_sys_instruction, instruction, instruction)

        with gr.Column(scale=2, variant="panel"):
            gr.Markdown("## Select the Generation Model")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    llm = gr.Dropdown(
                        choices=[
                            "VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge",
                            "vilm/vinallama-7b-chat",
                            "vilm/VinaLlama2-14B",
                            "vilm/vinallama-2.7b",
                        ],
                        value="vilm/vinallama-7b-chat",
                        label="Select the LLM",
                    )
                    do_quantize = gr.Checkbox(
                        label="Do quantization 8-bit",
                        info="Reduce memory by 2 (good for >10B models)",
                    )

                with gr.Column(scale=1):
                    model_load_btn = gr.Button("Load model", variant="primary", scale=1)
                    load_success_msg = gr.Textbox(
                        show_label=False, lines=1, placeholder="Model loading ..."
                    )
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Chatbox",
                height=600,
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

            model_load_btn.click(
                load_models,
                [embedding_model, llm, do_quantize],
                load_success_msg,
                api_name="load_models",
            )

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot,
                [
                    chatbot,
                    instruction,
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    top_k,
                    top_p,
                    k_context,
                ],
                chatbot,
            )
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot,
                [
                    chatbot,
                    instruction,
                    temperature,
                    max_new_tokens,
                    repetition_penalty,
                    top_k,
                    top_p,
                    k_context,
                ],
                chatbot,
            ).then(clear_cuda_cache, None, None)

            clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    load_dotenv()
    demo.queue()
    demo.launch(max_threads=3, debug=True, share=True)
