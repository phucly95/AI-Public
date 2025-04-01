## Gen AI - LLM, RAG & Fine-tuning – Hiểu và Ứng Dụng
Chào mọi người,

AI tạo sinh (Gen AI) đang ngày càng được ứng dụng rộng rãi, xu hướng phát triển của AI là không thể đảo ngược, vì thế chúng ta cần tìm cách sử dụng nó một cách hợp lý và hiệu quả hơn.

Với anh chị em DE chúng ta không những có thể sử dụng các công cụ AI có sẵn (chatgpt, gemini, awsQ, midjourney, kling ai...) mà còn có khả năng tự viết các phần mềm, ứng dụng của riêng mình sử dụng các model AI miễn phí (ví dụ deepseek, llama3, qwen ...) hoặc trả phí (gpt, claude...) và kiểm soát được các model đó với dữ liệu cá nhân của mình, phản hồi theo ngữ cảnh mà mình muốn.
Thế nên hôm nay em/mình muốn chia sẻ một chút về chủ đề Gen AI, cụ thể là LLM, RAG & Fine-tuning để mọi người cùng thảo luận, trao đổi kiến thức, cũng là cách em/mình verify lại những hiểu biết của mình về lĩnh vực này.

Hy vọng với những chia sẻ này có thể giúp anh chị em DE tiết kiệm thời gian, công sức tìm hiểu và tiếp cận lĩnh vực này một cách nhanh chóng.

Nếu có chỗ nào còn chưa đúng xin mọi người đóng góp ý kiến để em/mình sửa lại nhé.

## Có những giải pháp nào để có thể tạo phần mềm/ ứng dụng AI của riêng mình ?
1. Build model từ đầu (cần kiến thức chuyên môn và lượng tài nguyên cực lớn) => không phải mục tiêu của chúng ta
2. Finetuning model opensource có sẵn => Không cần quá nhiều tài nguyên, mất ít thời gian hơn, chạy trên gpu.
3. Áp dụng RAG(Retrieval-Augmented Generation) và tìm kiếm thông tin (retrieval) => Cần rất ít tài nguyên, có thể chạy bằng cpu, có thể cập nhật kiến thức mới liên tục.

Mỗi giải pháp trên đều có ưu và nhược điểm riêng, chúng ta sẽ so sánh sau khi đã hiểu về cách triển khai của từng giải pháp.
Để có thể thực hiện được các giải pháp trên, trước hết chúng ta cần hiểu thêm về LLM.

## Sơ lược quy trình suy luận (inference) của LLM
Hầu hết các model LLM hiện nay đều sử dụng kiến trúc transformer nên có dùng quy trình xử lý như sau:
#### Text/Prompt => Tokenize => Embedding => Attention & Transformer Layers => Decoding & Sampling => Output
Phần này hơi lý thuyết nhưng em/mình sẽ cố gắng diễn giải ngắn gọn để mọi người hiểu những bước quan trọng để khi finetuning hoặc áp dụng RAG vào model LLM sẽ cần đến.

## Giải thích các khái niệm ở trên và những gì xảy ra ở bước đó
- LLM viết tắt của Large Language Model là các mô hình ngôn ngữ lớn (hàng trăm triệu đến hàng tỉ tham số) được huấn luyện trên tập dữ liệu rất lớn (hàng nghìn tỉ token) trong nhiều giờ bằng rất nhiều gpu. Nhận đầu vào là một chuỗi text và trả ra kết quả có thể là chuỗi text có liên quan đến input (làm thơ, trả lời yêu cầu ...) hoặc xác suất (đối với bài toán phân lớp).

- Text/Prompt: là chuỗi các yêu cầu người dùng ví dụ: "Hôm nay là thứ mấy?"

- Tokenize: Là quá trình tách từ và mã hoá từ thành các con số (gọi là các token), ví dụ "Hôm nay là thứ mấy?" sẽ được tách thành ["Hôm", "nay", "là", "thứ", "mấy", "?"] sau đó số hoá thành [22,42,11,23,65,73]. Việc tokenize sẽ đảm bảo với mỗi con số tương ứng với 1 từ duy nhất. Tổng tất cả bộ tokens của model gọi là Vocabulary(Vocab).

- Embedding là quá trình ánh xạ các token đã mã hoá vào không gian vector N chiều. Sau khi embedding các từ có tính chất giống nhau sẽ được ở gần nhau trong không gian vector đó ("đẹp" sẽ ở gần "xinh" và cách xa "xấu"), ví dụ Vector của token "AI" có thể là [0.23, -0.67, 0.89, ..., 0.12]  (768 chiều với GPT-3.5). Vì tính chất này nên embedding sẽ được dùng đễ vector hoá thông tin document trong RAG sau đó sẽ tìm kiếm ngữ cảnh dựa vào độ tương đồng giữa 2 vector query và document (similarity search).

- Attention & Transformer Layers: Phần này khá dài, phức tạp và không quá quan trọng để có thể áp dụng RAG hay finetuning nên mọi người có thể tìm hiểu thêm tại https://www.youtube.com/watch?v=_Zt23FA31co&t=125s

- Decoding & Sampling:  ở bước này sẽ cấu hình để LLM sáng tạo hơn (temperature), Giới hạn số lượng từ có xác suất cao nhất để lựa chọn(Top-k Sampling) ... phần này chủ yếu để cấu hình model khi sử dụng.

- Output: Các token sinh ra lần lượt dưới dạng số được giải mã (decode) về text và ghép lại với nhau thành câu trả lời. Việc sinh từ xảy ra tuần tự như sau:

=> Người dùng input vào đoạn text "Hôm nay là thứ mấy?"

=> model sinh ra token 31 tương đương với từ "Thứ"

=> model ghép từ mới sinh với input trước đó thành "Hôm nay là thứ mấy? Thứ" để làm input suy luận từ tiếp theo.

=> model sinh ra token 29 tương đương với từ "ba".

=> lặp lại cho đến khi model dự đoán token tiếp theo là token <end> (hoặc một token nào đó khác có ý nghĩa tương tự tuỳ vocab của model) thì model sẽ dừng lại. Nếu input đạt đến ngưỡng tối đa về token size (số lượng token input cùng lúc vào cho model) thì model cũng sẽ dừng generate token mới.

Sau đó ghép chuỗi có được lại và cắt đi phần input ban đầu ta được câu trả lời "Thứ ba".

### Note: Từ việc hiểu được luồng suy luận của LLM chúng ta có thể rút ra một số lưu ý như sau:
- Model sẽ sinh thêm từng từ một chứ không thể sinh tất cả cùng lúc.
- Text/Prompt input vào cho LLM model có giới hạn về length, nếu chúng ta input đoạn text quá dài có thể vượt quá giới hạn của model gây lỗi hoặc bị cắt bớt nội dung của Text/Prompt.
- Việc input dài cho một câu hỏi sẽ làm giới hạn về token size đạt đến sớm, vì thế hãy cố gắng input ngắn gọn và đủ ý, loại bỏ các ký tự thừa trong input trước khi đưa vào LLM suy luận.
- Tương tự thì Embedding cũng có giới hạn về token size. Cần chú ý khi sử dụng RAG

# Fine-tuning LLM
- LLM là neural network rất lớn với rất nhiều layer và số lượng parameters lên đến hàng tỉ vì thế việc training với tất cả lượng parameters đó là rất tốn kém. Người ta đã chứng minh được rằng, các lớp càng gần input thì càng mang tính tổng quát, đặc trưng chung, các lớp càng gần output thì càng mang tính riêng biệt cho các bài toán cụ thể. Vậy nên kĩ thuật finetuning ra đời, bằng cách loại bỏ đi những layer gần output của model, thay chúng bằng các layer mới và chỉ cập nhật lại các parameters của các layer mới đó giúp tiết kiệm chi phí tính toán rất nhiều.

=> Tiết kiệm đáng kể chi phí tính toán vì chỉ cần cập nhật một phần nhỏ của mô hình.

=> Giữ lại kiến thức chung của mô hình gốc, nhưng tinh chỉnh để phù hợp với tác vụ cụ thể.

=> Huấn luyện nhanh hơn mà vẫn duy trì hiệu suất tốt trên dữ liệu mới.

Note: Finetuning là kỹ thuật có thể áp dụng với hầu hết neural network models không chỉ mỗi LLM

- Với LLM người ta thường sử dụng LORA(Low-Rank Adaptation) để finetuning model. Bằng cách bổ sung thêm ma trận có hạng thấp (low rank) vào phần output của model sau đó tinh chỉnh tham số của ma trận đó và giữ nguyên model gốc (không cần bỏ đi những layer ở gần cuối) giúp giảm chi phí tính toán, tiết kiệm bộ nhớ, tăng tốc độ huấn luyện model đồng thời có thể lưu ma trận đó lại và chia sẻ với nhau.

# RAG
Các mô hình LLM như GPT-4, LLaMA, hay Deepseek-R1 có một hạn chế lớn:
- Không thể cập nhật kiến thức sau khi huấn luyện.

- Bị giới hạn trong phạm vi dữ liệu mà chúng đã học.

- Tốn kém nếu muốn fine-tune lại với dữ liệu mới.

Giải pháp cho các hạn chế đó là sử dụng RAG (Retrieval-Augmented Generation):
- Thay vì lưu trữ toàn bộ kiến thức trong bộ nhớ của LLM, ta dùng một kho dữ liệu ngoài (vector database) để lưu trữ tài liệu.

- Khi nhận được câu hỏi, hệ thống sẽ tìm kiếm thông tin từ kho dữ liệu, sau đó kết hợp với LLM để tạo ra câu trả lời chính xác hơn.

Quy trình hoạt động của RAG
1. Nhận input từ người dùng (Prompt)

Ví dụ: "Tóm tắt bài báo khoa học về LLM RAG?"

2. Tokenize & Embed prompt

Câu hỏi của người dùng được chuyển thành vector embedding.

3. Truy vấn kho dữ liệu (Vector Store Search)

Hệ thống tìm kiếm tài liệu liên quan trong kho dữ liệu vector (ChromaDB, Pinecone, FAISS…).

Ví dụ: Trả về một đoạn tài liệu có nội dung liên quan đến "LLM RAG".

4. Kết hợp dữ liệu & LLM (Augment & Generate)

LLM sẽ sử dụng tài liệu tìm được để trả lời câu hỏi chính xác hơn.

Ví dụ: "Dựa trên bài báo khoa học, LLM RAG là một mô hình kết hợp giữa…".

5. Trả kết quả cho người dùng

Người dùng nhận được câu trả lời đúng, cập nhật và chi tiết hơn so với mô hình LLM thông thường.

# So sánh giữa Fine-tuning và RAG

| **Tiêu chí**         | **Fine-tuning** | **RAG** |
|----------------------|----------------|---------|
| **Mục đích**        | Điều chỉnh mô hình để học từ dữ liệu mới | Kết hợp tìm kiếm thông tin với LLM để trả lời chính xác hơn |
| **Cách hoạt động**  | Huấn luyện lại mô hình trên dữ liệu cụ thể | Tìm kiếm thông tin trong cơ sở dữ liệu vector và kết hợp với LLM |
| **Cập nhật dữ liệu mới** | ❌ Cần huấn luyện lại | ✅ Chỉ cần cập nhật dữ liệu trong vector store |
| **Chi phí tài nguyên** | 🔴 Cao (cần GPU mạnh) | 🟢 Thấp (chỉ cần vector database) |
| **Tốc độ triển khai** | 🔴 Chậm (vài giờ đến vài ngày) | 🟢 Nhanh (chỉ vài giây để thêm tài liệu mới) |
| **Tính linh hoạt** | 🟡 Chỉ giỏi trong phạm vi dữ liệu đã fine-tune | 🟢 Có thể cập nhật thông tin mới liên tục |
| **Ứng dụng phù hợp** | Khi cần điều chỉnh phong cách hoặc chuyên sâu về một lĩnh vực | Khi cần cập nhật dữ liệu mới và tìm kiếm thông tin theo ngữ cảnh |
| **Ví dụ sử dụng** | Dạy GPT cách viết theo phong cách của công ty | Hỏi đáp dựa trên tài liệu nội bộ |

# Code example
Sau một chuỗi lý thuyết đau đầu thì cũng đến lúc chúng ta tự tạo Gen AI của riêng mình :D

Dưới đây là ví dụ về việc Finetuning LLM và kết hợp RAG với LLM sử dụng ngôn ngữ python.

### Tại sao lại là python ?
- Cú pháp đơn giản, dễ đọc, viết.
- Thư viện AI cực lớn giúp giảm thời gian code, cộng đồng lớn hỗ trợ giải quyết các vấn đề gặp phải.
- Tận dụng tốt GPU bằng các thư viện ví dụ tensorflow, pytorch...
- Và một điều quan trọng nữa là code python rất ngắn gọn sẽ hiệu quả khi generate code bằng các công cụ hoặc model AI (copilot, gpt, gemini ...).

### Open AI(gpt4-o) + RAG (cần có open_ai_api_key)
#### open_ai_api_key có thể lấy bằng việc đăng ký tài khoản và thanh toán tại https://platform.openai.com/

Clone project hoặc download trực tiếp file này về máy:
https://github.com/phucly95/AI-Public/blob/main/openai-rag.py
Để chạy được code này cần cài python 3.12 và sau đó chạy thử project, nếu báo not found library nào thì cài cái đó là xong :D
```
python3 ./openai-rag.py # chỉnh lại file path nếu cần
```
### Deepseek R1 + RAG (Model opensource nhưng cần máy ram > 16gb để chạy được).
- Để chạy được code này cần deeploy deepseek để làm LLM. Cách dễ dàng nhất là cài Ollama tại https://ollama.com/download
- Sau khi cài Ollama xong mở terminal và chạy lệnh:
```
ollama run deepseek-r1:1.5b
```
Chờ cho model download và chạy xong thì model deepseek đã được deploy lên http://localhost:11434
- Clone project hoặc download trực tiếp file này về máy:
https://github.com/phucly95/AI-Public/blob/main/deepseek-rag.py
- Chạy thử và cài các thư viện còn thiếu
```
python3 ./deepseek-rag.py
```

### Finetuning-LLM
...


Do bài viết cũng đã dài nên em/mình xin phép được để lại phần này trong bài viết riêng sau. Hy vọng sẽ có những đóng góp của các anh chị em có kinh nghiệm về lĩnh vực này để giúp em cải thiện những kiến thức hổng, những chỗ còn thiếu xót hoặc hiểu sai. Em xin cảm ơn !

# AI Agent & Langchain
Một chủ đề cũng rất hot gần đây mà em muốn chia sẻ cùng mọi người, chắc hẳn cũng nhiều anh chị em nghe đến AI Agent, Cursor, Manus ...

Vậy AI Agent là gì ? Ứng dụng như thế nào và làm thế nào để tự tạo các AI Agent của riêng mình?

Các model opensource hiện nay có sử dụng làm AI Agent được không ? Và sử dụng như thế nào ?

Anh chị em DE đã sử dụng AI Agent nào chưa? cùng thảo luận ở đây nhé !
