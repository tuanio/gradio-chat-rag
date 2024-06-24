from langchain_core.prompts.prompt import PromptTemplate

_template = """Cung cấp lịch sử trò chuyện và câu hỏi tiếp theo của người dùng, hãy tóm tắt và đưa ra câu hỏi duy nhất cô đọng toàn bộ nội dung.

Lịch sử:
{chat_history}
Câu hỏi tiếp theo: {question}
Câu hỏi cô đọng:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
