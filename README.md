# Tạo sinh tăng cường truy xuất - Retrieval Augment Generation

Kho lưu trữ này chứa mã nguồn và tài nguyên liên quan đến Tạo sinh tăng cường truy xuất (RAG), một kỹ thuật được thiết kế để giải quyết vấn đề dữ liệu cũ trong các Mô hình Ngôn ngữ Lớn (LLM) như Llama-2. Các LLM thường thiếu nhận thức về các sự kiện gần đây và thông tin cập nhật. RAG kết hợp kiến thức bên ngoài từ cơ sở kiến thức vào các phản hồi của LLM, cho phép tạo ra các câu trả lời chính xác và có cơ sở vững chắc.

## Nội dung kho lưu trữ
- `src`: Chứa mã nguồn để triển khai kỹ thuật RAG và tương tác với cơ sở kiến thức.
- `data`: Lưu trữ bộ dữ liệu và tài nguyên liên quan để xây dựng cơ sở kiến thức.
- `db`: Để quản lý và lưu trữ nhúng token hoặc biểu diễn vector cho tìm kiếm cơ sở kiến thức.
- `requirements.txt`: Các gói Python cần thiết để chạy mã trong kho lưu trữ này.

## Về RAG (Tạo sinh tăng cường truy xuất)
RAG là một phương pháp mới kết hợp khả năng của các Mô hình Ngôn ngữ Lớn (LLM) với các cơ sở kiến thức bên ngoài để nâng cao chất lượng và độ mới của các phản hồi được tạo ra. Nó giải quyết thách thức về thông tin lỗi thời bằng cách truy xuất kiến thức liên quan theo ngữ cảnh từ các nguồn bên ngoài và kết hợp nó vào nội dung được tạo ra bởi LLM.

## Về Gradio
[Gradio](https://www.gradio.app) là một thư viện Python giúp bạn nhanh chóng tạo giao diện người dùng cho các mô hình học máy của mình. Nó cho phép bạn triển khai nhanh chóng các mô hình và làm cho chúng dễ tiếp cận thông qua một giao diện thân thiện với người dùng mà không cần phát triển frontend phức tạp. Một ứng dụng Gradio được khởi chạy khi mã `gradio_chatbot.py` được chạy. Nó chứa các phần tử có thể điều chỉnh như mô hình Nhúng, mô hình Tạo sinh, lời nhắc hệ thống có thể chỉnh sửa, và các tham số có thể điều chỉnh của LLM đã chọn.

### Các bước
Để sử dụng mã trong kho lưu trữ này, hãy làm theo các bước sau:
1. Clone kho lưu trữ về máy local của bạn.
2. Di chuyển đến thư mục kho lưu trữ bằng dòng lệnh.
3. Cài đặt các gói cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

4. Chạy ứng dụng chatbot bằng lệnh:

```bash
python src/gradio_chatbot.py
```

5. Khi ứng dụng Gradio đã chạy, tải lên một tài liệu (pdf hoặc csv), chọn các mô hình (nhúng và tạo sinh), điều chỉnh các tham số có thể điều chỉnh, chỉnh sửa lời nhắc hệ thống, và hỏi bất cứ điều gì bạn cần!

# Steps

1. Tạo một file pdf dữ liệu đặt tên là "chatbot_knowledge.pdf", chứa các nội dung muốn chatbot đọc và trả lời, các câu hỏi của người dùng sẽ dựa vào nội dung đó để hỏi đáp.
2. Load Model LLM & model embedding - có thể upload thêm file để load cùng dữ liệu "chatbot_knowledge.pdf"


## Nâng tầm hệ thống hỏi đáp tiếng Việt với RAG tiên tiến

**Mô hình tiên tiến:**

Hệ thống được trang bị bộ đôi mô hình mạnh mẽ:

* **Embedding model:** Sử dụng [Fsoft-AIC/videberta-base](https://huggingface.co/Fsoft-AIC/videberta-base), mô hình ngôn ngữ BERT tiếng Việt do FPT AI phát triển, vượt trội PhoBERT - từng là mô hình tiếng Việt tốt nhất. Model này tạo vector biểu diễn cho các đoạn văn bản, giúp truy vấn thông tin chính xác dựa trên ngữ cảnh câu hỏi.
* **LLM:** Vinallama-7b-chat từ [vilm/vinallama-7b-chat](https://huggingface.co/vilm/vinallama-7b-chat), phiên bản tiếng Việt của LLama do **vilm** phát triển, paper gốc [paper VinaLLaMA](https://arxiv.org/abs/2312.11011). Model này sinh văn bản, trả lời câu hỏi dựa trên nội dung được cung cấp. 

**Khả năng vượt trội:**

* **Hiểu tiếng Việt sâu sắc:** Nhờ sự kết hợp hoàn hảo giữa Fsoft-AIC/videberta-base và Vinallama-7b-chat, hệ thống có khả năng hiểu ngôn ngữ tiếng Việt một cách sâu sắc, nắm bắt chính xác ý đồ người dùng.
* **Trả lời câu hỏi toàn diện:** Hệ thống có thể truy cập và xử lý thông tin từ nhiều nguồn khác nhau, cung cấp câu trả lời đầy đủ, chính xác và chi tiết cho mọi câu hỏi của người dùng.
* **Hỗ trợ đa dạng:** Hệ thống có thể hỗ trợ nhiều tác vụ khác nhau như:
    * Tóm tắt văn bản
    * Dịch thuật
    * Viết các loại văn bản sáng tạo
    * Và nhiều hơn nữa

**Khả năng mở rộng:**

Hệ thống được thiết kế linh hoạt, cho phép người dùng dễ dàng huấn luyện thêm mô hình trên dữ liệu chuyên ngành để đáp ứng nhu cầu cụ thể. Tuy nhiên, một người dùng mới hoàn toàn có thể chỉ cần thêm dữ liệu vào "chatbot_knowledge.pdf" là hệ thống sẽ hoạt động tốt.