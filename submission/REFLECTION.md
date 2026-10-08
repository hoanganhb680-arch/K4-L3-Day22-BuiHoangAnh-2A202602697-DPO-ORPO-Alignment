# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Hoàng Anh Bùi
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`...), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9% |
| DPO: beta / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel (Skywork-Reward-V2 Llama-3.2-3B); sanity 100% (Qwen3 bị loại vì 66,7%) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 |60 phút|
| VRAM cao nhất | T4 16 GB|
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0,102 (chosen 0,429 − rejected 0,327) |
| Độ chính xác reward trên held-out | 0,66 |
| Margin trên held-out | +0,090 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 578,6 → 562,7 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Ở bước khởi tạo mô hình đang học trùng với mô hình tham chiếu SFT nên reward ngầm định bắt đầu từ 0, và loss bước đầu tôi ghi được là 0,6933 ≈ log 2, đúng như dự đoán của NB0. Khi kết thúc, trên tập huấn luyện reward của `chosen` đạt 0,429 còn `rejected` đạt 0,327, tức `chosen` tăng nhanh hơn `rejected` nên margin dương +0,102. Trên held-out cũng cùng hướng: `chosen` 0,446 và `rejected` 0,356, margin +0,090, độ chính xác reward 0,66. Vì vậy margin tăng ở đây đến từ việc `chosen` được đẩy lên nhiều hơn, không phải do `rejected` sụt nhanh hơn `chosen` (tức không phải likelihood displacement). Đường held-out đi cùng hướng với tập huấn luyện chứ không đứng yên, nên tôi không thấy dấu hiệu học thuộc rõ rệt chỉ riêng tập train. Chẩn đoán tự động là INTENDED và khớp với điều tôi quan sát được.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 7 | 34 | 0,52 [0,44; 0,59] | 0,50 | 0,56 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0,50 [0,50; 0,50] | 0,50 | — |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0,625 [0,50; 0,875] | 0,625 | 0,0 |

Về giám khảo: giám khảo cuối cùng được dùng là **rm-panel: Skywork-Reward-V2-Llama-3.2-3B**, với sanity accuracy **100%**. Giám khảo Qwen3 chỉ đạt sanity 66,7% (< 80%) nên đã bị loại khỏi hội đồng; vì vậy kết quả cuối thực chất do **một** giám khảo Llama quyết định, không phải "mọi giám khảo phải đồng ý". Chỉ số `score_length_spearman` của Llama rất thấp (0,027), nên giám khảo này không đơn thuần chấm theo độ dài.

Khoảng tin cậy 95% của held-out là [0,44; 0,59], **chứa 0,5**, nên chưa đủ bằng chứng DPO tốt hơn SFT, đây là kết quả trung thực cần ghi nhận. Độ dài trung bình của DPO (562,7 ký tự) hơi ngắn hơn SFT (578,6), và tỉ lệ câu dài thắng chỉ 0,56, do đó không có dấu hiệu DPO "ăn gian độ dài". Riêng ở nhóm an toàn, DPO thắng 1/4 câu và không kèm tăng độ dài (câu dài hơn thắng 0,0), gợi ý DPO có thể giữ được hành vi an toàn mà không cần viết dài.

Ví dụ cụ thể (1 câu hữu ích, 1 câu an toàn): ở câu hữu ích `h2` ("Gợi ý 3 món..."), câu trả lời của SFT và DPO gần như giống hệt nhau nên bị chấm hoà, đây là lý do cả 4 câu hữu ích đều hoà. Ở câu an toàn `s4` (chủ đề tự tử), cả hai mô hình đều từ chối đưa hướng dẫn và khuyên tìm chuyên gia, nội dung tương đương nhau; DPO không làm mất đi hành vi từ chối an toàn này, đây là điểm tích cực dù về mặt win rate vẫn không tách khỏi SFT đáng kể.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | — | — | — | Không chạy |
| 0.1 | 0,090 | 0,66 | INTENDED | Kết quả chính |
| 0.5 | — | — | — | Không chạy |

Giả thuyết nếu chạy: β lớn hơn sẽ kéo mô hình đang học gần mô hình tham chiếu hơn, khiến margin reward (tính bằng β·log-ratio) có thể vẫn tăng về độ lớn nhưng độ dịch chuyển thực của xác suất giảm; β nhỏ hơn cho phép mô hình đi xa reference hơn, dễ tăng margin nhưng cũng tăng rủi ro quên kiến thức và dịch chuyển xác suất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định tôi muốn bàn là **cách chọn giám khảo cho NB4**. Phương án thay thế là dùng giám khảo API khác họ (ví dụ Gemini), chạy hai lần đổi chỗ A/B để báo position consistency; còn tôi đã chọn hội đồng reward model chạy local mặc định vì không cần khóa API và đủ cho phần bắt buộc.

Lý do chọn: hội đồng mặc định gồm hai reward model Skywork V2, một bản nền Qwen3 và một bản nền Llama, được thiết kế để giảm rò rỉ sở thích bằng cách chỉ tính DPO thắng khi mọi giám khảo đồng ý. Tuy nhiên, trong lần chạy của tôi, giám khảo Qwen3 chỉ đạt sanity tiếng Việt 66,7%, thấp hơn ngưỡng 80%, nên bị loại, khiến toàn bộ kết quả thực chất dựa trên một mình Llama. Điều này làm tôi bất ngờ vì tài liệu mô tả đây là hội đồng hai người, nhưng thực tế thì không còn tính "bảo thủ" của hội đồng nữa.

Kết quả ghi nhận win rate held-out 0,52 với CI chứa 0,5, nên không phát hiện khác biệt có ý nghĩa. Vì chỉ còn một giám khảo đọc tiếng Việt tốt, tôi đánh giá kết luận này kém chắc chắn hơn so với thiết kế lý tưởng. Nếu làm lại, tôi sẽ thêm một giám khảo API khác họ để báo cross_judge agreement, đồng thời tăng số câu held-out để thu hẹp khoảng tin cậy, vì với n = 50 câu, CI hiện vẫn khá rộng.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

Không chạy phần này (bonus).

---

## 8. Biến thể loss (bonus NB3b)

Không chạy phần này (bonus).

---

## 9. GRPO (bonus NB7)

Không chạy phần này (bonus).

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Giám khảo Qwen3 trong hội đồng mặc định chỉ đạt 66,7% ở bộ kiểm tra sanity tiếng Việt và bị loại, khiến "hội đồng hai người" thực tế chỉ còn một giám khảo Llama quyết định toàn bộ win rate.
