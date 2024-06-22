from torch import cuda, bfloat16
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login
from langchain.prompts import PromptTemplate
from util.utils import load_embedding_model
from util.prompts import CONDENSE_QUESTION_PROMPT


class Conversation_RAG:
    def __init__(
        self,
        embedding_model_repo_id="sentence-transformers/all-roberta-large-v1",
        llm_repo_id="meta-llama/Llama-2-7b-chat-hf",
        hf_token="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",
    ):
        self.hf_token = hf_token
        self.embedding_model_repo_id = embedding_model_repo_id
        self.llm_repo_id = llm_repo_id

    def load_model_and_tokenizer(self, do_quantize=False):
        embedding_model = load_embedding_model(self.embedding_model_repo_id)
        vectordb = FAISS.load_local(
            "./db/faiss_index", embedding_model, allow_dangerous_deserialization=True
        )

        if self.hf_token:
            login(token=self.hf_token)

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        # device = 'mps'

        if do_quantize:
            # bnb_config = transformers.BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type='nf4',
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=bfloat16
            # )

            bnb_config = transformers.BitsAndBytesConfig(
                load_in_8bit=True,
            )

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.llm_repo_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map='auto',
                torch_dtype=bfloat16,
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.llm_repo_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=bfloat16,
            )
        model.eval()

        print("Device:", model.device)

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_repo_id)
        return model, tokenizer, vectordb

    def create_conversation(
        self,
        model,
        tokenizer,
        vectordb,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.1,
        top_k=20,
        top_p=0.95,
        k_context=20,
        num_return_sequences=1,
        instruction="Bạn là trợ lý ảo thông minh. Bạn sẽ đọc nội dung từ ngữ cảnh để trả lời câu hỏi của người dùng. Trả lời ngắn gọn, đủ ý, nội dung bằng tiếng Việt.",
    ):  
        streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,  # langchain expects the full text
            task="text-generation",
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            streamer=streamer,
            torch_dtype=bfloat16,
            eos_token_id=tokenizer.eos_token_id
        )

        condense_gen_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,  # langchain expects the full text
            task="text-generation",
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            top_k=top_k,
            top_p=top_p,
            torch_dtype=bfloat16,
            num_return_sequences=num_return_sequences,
            eos_token_id=tokenizer.eos_token_id
        )

        llm = HuggingFacePipeline(pipeline=generate_text)
        condense_llm = HuggingFacePipeline(pipeline=condense_gen_text)

        # system_instruction = f"Người dùng: {instruction}\n"
        # template = (
        #     system_instruction
        #     + """
        # Ngữ cảnh:\n
        # {context}\n
        # Câu hỏi: {question}\n
        # Trợ lý:
        # """
        # )

        template = f"<|im_start|>system: {instruction}<|im_end|>"
        template += """<|im_start|>
        user: {context}
        question: {question}
        <|im_end|>
        <|im_start|>
        assistant:"""

        QCA_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=template
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
            combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            get_chat_history=lambda h: h,
            condense_question_llm=condense_llm,
            verbose=True,
        )

        if streamer:
            return qa, streamer

        return qa