# Lab 21 — LoRA Fine-tuning Evaluation Report

**Học viên**: Vu Le Hoang  
**MSSV**: 2A202600342  
**Submission option**: B (HF Hub)  
**Ngày nộp**: 2026-05-07

## 1. Setup

- **Base model**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 500 samples total
- **Split**: 450 train / 50 eval (90/10, seed = 42)
- **GPU**: A100 40GB
- **HF Hub link**: https://huggingface.co/itsmehonga/lab21-llama-3.1-8b-vi-r16
- **Quantization**: QLoRA 4-bit NF4 with Unsloth
- **LoRA baseline**: `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj"]`
- **Stretch experiment**: `r=16` with all projection layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
- **max_seq_length**: chosen from token-length p95 with a cap at 2048 in the notebook preprocessing step
- **Estimated training cost**: about **$0.17** total for the 3 rank runs + stretch on A100 Colab Pro pricing (`$1.18/hr`), based on the recorded training times
- **Submission folder**: `lab21_2A202600342/`
- **HF Hub link**: https://huggingface.co/itsmehonga/lab21-llama-3.1-8b-vi-r16

### Notes from the notebook

The notebook follows the rubric’s Alpaca-style formatting and evaluates the model in both quantitative and qualitative ways. For submission, I keep only the stripped notebook plus the CSV artifacts required by Option B.

### Submission contents

- `lab21_2A202600342/REPORT.md`
- `lab21_2A202600342/notebook.ipynb`
- `lab21_2A202600342/results/rank_experiment_summary.csv`
- `lab21_2A202600342/results/qualitative_comparison.csv`
- `lab21_2A202600342/LINKS.md`

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| Base | 0 | N/A | N/A | 1.6074 | 4.9898 |
| 8 | 3,407,872 | 2.00 min | 20.59 GB | 1.3680 | 3.9273 |
| 16 | 6,815,744 | 2.17 min | 19.58 GB | 1.3615 | 3.9021 |
| 64 | 27,262,976 | 2.02 min | 21.95 GB | 1.3563 | 3.8820 |
| 16-ALL | 41,943,040 | 2.30 min | 21.31 GB | 1.3644 | 3.9132 |

### Quick read

- All LoRA variants improved substantially over the base model: perplexity dropped from **4.99** to about **3.88–3.93**.
- `r=64` gave the best perplexity in this run, but the gain over `r=16` was very small.
- `r=16-ALL` increased capacity a lot, but it did not beat the baseline `r=16` on this dataset.

## 3. Loss Curve Analysis

The loss curve is stable and does not show strong overfitting. Training loss falls from roughly **1.67** to **1.26** over the run, while eval loss stays in a narrow band around **1.30–1.36**.

What I observe:

- The train loss decreases steadily, with normal minibatch noise.
- Eval loss is mostly flat and slightly downward overall, which suggests the model is learning the dataset without memorizing it aggressively.
- The final eval point is a bit higher than the previous one, but that looks like late-epoch variance rather than a clear overfitting trend.

Overall, this is a healthy fine-tuning curve for a 500-sample instruction dataset on an A100 profile.

## 4. Qualitative Comparison

The notebook generated 10 prompts; below are 5 representative examples from `qualitative_comparison.csv`.

| Prompt | Base model observation | Fine-tuned (r=16) observation |
|--------|------------------------|--------------------------------|
| Giải thích khái niệm machine learning cho người mới bắt đầu | Trả lời đúng hướng nhưng khá chung chung, mô tả theo kiểu giải thích đại cương | Câu trả lời có cấu trúc hơn, dùng ngôn ngữ Việt tự nhiên hơn và đi thẳng vào định nghĩa |
| Viết đoạn code Python tính số Fibonacci thứ n | Có code mẫu cơ bản, nhưng phần giải thích còn dài và hơi vòng | Câu trả lời ngắn gọn hơn, code rõ hơn và ít lan man hơn |
| Liệt kê 5 nguyên tắc thiết kế UI/UX | Đưa ra các ý khá chung, thiếu nhấn mạnh vào tính hệ thống | Câu trả lời trình bày theo danh sách rõ ràng và dễ đọc hơn |
| Tóm tắt sự khác biệt giữa LoRA và QLoRA | Có nêu ý đúng nhưng diễn đạt chưa sắc nét, đôi khi lẫn khái niệm | Phân biệt rõ hơn giữa adapter-based fine-tuning và quantized base model |
| Phân biệt prompt engineering, RAG, và fine-tuning | Mô tả ba khái niệm nhưng chưa tách vai trò rõ | So sánh có trật tự hơn, phân loại đúng use case của từng kỹ thuật |

### Nhận xét ngắn

Fine-tuned model cải thiện chủ yếu ở **độ có cấu trúc**, **độ nhất quán**, và **phong cách trả lời**. Mức cải thiện kiến thức không quá lớn, điều này hợp lý vì dataset là instruction dataset tổng quát tiếng Việt, không phải một domain hẹp.

## 5. Conclusion về Rank Trade-off

Kết luận thực tế của run này là `r=16` cho ROI tốt nhất. Về chất lượng, `r=64` đạt perplexity thấp nhất trong bảng, nhưng mức cải thiện so với `r=16` rất nhỏ: chỉ khoảng 0.5% tương đối, trong khi số tham số trainable tăng gấp 4 lần và peak VRAM cũng cao hơn. Nếu mục tiêu là tối ưu chất lượng tuyệt đối, `r=64` có thể là lựa chọn cuối cùng, nhưng trên dataset 500 mẫu này, phần lợi ích thêm không đủ lớn để bù cho chi phí bộ nhớ và độ phức tạp cao hơn.

`r=8` là phương án nhẹ nhất về mặt adapter size và vẫn cải thiện đáng kể so với base model, nên nó là lựa chọn hợp lý khi cần tiết kiệm tài nguyên. Tuy vậy, `r=16` cân bằng tốt nhất giữa hiệu năng, độ ổn định và chi phí. Nó cho perplexity gần sát `r=64`, nhưng chỉ dùng khoảng một phần tư số tham số trainable của `r=64`. Thêm vào đó, curve loss của `r=16` cũng ổn định, không cho thấy vấn đề overfitting đáng kể. Với bài toán instruction tuning tiếng Việt kiểu này, tôi sẽ chọn `r=16` làm default để deploy hoặc nộp bài, còn `r=64` chỉ nên dùng khi thật sự cần squeeze thêm chất lượng và GPU budget cho phép.

## 6. What I Learned

- Cùng một dataset nhỏ, rank lớn hơn không luôn cho cải thiện tương xứng; diminishing returns xuất hiện rất sớm.
- `exp(eval_loss)` là một chỉ số dễ diễn giải để so sánh các adapter, nhưng cần kết hợp với qualitative output để tránh chỉ nhìn số mà bỏ qua hành vi sinh văn bản.
- Với QLoRA, phần khó nhất không phải load model mà là giữ pipeline ổn định giữa training, evaluation, và artifact saving trên GPU giới hạn.

## Appendix: Submission Files

- `results/rank_experiment_summary.csv`
- `results/qualitative_comparison.csv`
- `notebook.ipynb` (stripped outputs)
- `LINKS.md`
