# Text Analysis and Distribution Report

## 1. Text Metric Quantiles
| metric_label | text_scope | minimum | maximum | mean | median | standard_deviation | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Character count | clinical_report (INPUT) | 73.0 | 137.0 | 105.95 | 106.0 | 10.2 | 98.0 | 106.0 | 113.0 | 119.0 | 123.0 | 129.0 |
| Character count | summary (TARGET) | 42.0 | 210.0 | 162.68 | 171.0 | 26.2 | 154.0 | 171.0 | 180.0 | 187.0 | 191.0 | 197.52 |
| Word count | clinical_report (INPUT) | 12.0 | 23.0 | 17.52 | 18.0 | 1.69 | 17.0 | 18.0 | 19.0 | 20.0 | 20.0 | 21.0 |
| Word count | summary (TARGET) | 6.0 | 32.0 | 25.12 | 26.0 | 4.56 | 24.0 | 26.0 | 29.0 | 30.0 | 30.0 | 32.0 |
| Sentence count | clinical_report (INPUT) | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Sentence count | summary (TARGET) | 1.0 | 2.0 | 1.99 | 2.0 | 0.09 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| Unique word count | clinical_report (INPUT) | 12.0 | 22.0 | 17.33 | 17.0 | 1.7 | 16.0 | 17.0 | 18.0 | 19.0 | 20.0 | 21.0 |
| Unique word count | summary (TARGET) | 6.0 | 30.0 | 23.13 | 24.0 | 3.96 | 21.0 | 24.0 | 26.0 | 27.0 | 28.0 | 29.0 |
| Average word length | clinical_report (INPUT) | 3.65 | 6.76 | 5.1 | 5.1 | 0.54 | 4.68 | 5.1 | 5.5 | 5.82 | 6.0 | 6.25 |
| Average word length | summary (TARGET) | 4.27 | 6.95 | 5.47 | 5.42 | 0.38 | 5.19 | 5.42 | 5.72 | 6.0 | 6.13 | 6.5 |
| Digit count | clinical_report (INPUT) | 0.0 | 6.0 | 3.3 | 3.0 | 1.7 | 2.0 | 3.0 | 5.0 | 6.0 | 6.0 | 6.0 |
| Digit count | summary (TARGET) | 0.0 | 6.0 | 2.9 | 3.0 | 1.78 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 6.0 |
| Uppercase count | clinical_report (INPUT) | 1.0 | 11.0 | 5.41 | 5.0 | 2.58 | 4.0 | 5.0 | 8.0 | 8.0 | 9.0 | 11.0 |
| Uppercase count | summary (TARGET) | 1.0 | 12.0 | 7.21 | 7.0 | 2.49 | 6.0 | 7.0 | 9.0 | 9.0 | 12.0 | 12.0 |
| Lowercase count | clinical_report (INPUT) | 47.0 | 113.0 | 80.12 | 80.0 | 9.87 | 73.0 | 80.0 | 87.0 | 93.0 | 97.0 | 103.52 |
| Lowercase count | summary (TARGET) | 35.0 | 165.0 | 126.3 | 132.0 | 20.11 | 122.0 | 132.0 | 139.0 | 145.0 | 148.0 | 154.0 |
| Punctuation count | clinical_report (INPUT) | 0.0 | 4.0 | 1.35 | 1.0 | 0.89 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 |
| Punctuation count | summary (TARGET) | 1.0 | 4.0 | 2.43 | 2.0 | 0.56 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 |

## 2. Vocabulary & NLP Characteristics
| text_scope | total_tokens | unique_tokens_vocab_size | type_token_ratio_ttr | hapax_legomena_count | hapax_percentage | average_sentence_length_words | average_word_length_chars | top_frequent_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_report (INPUT) | 82824 | 132 | 0.0016 | 0 | 0.0 | 17.52 | 5.1 | pt (3398), ae (2862), mg (2497), mutation (1885), dose (1828) |
| summary (TARGET) | 119318 | 96 | 0.0008 | 0 | 0.0 | 12.6 | 5.43 | with (7976), a (7323), reported (5258), an (4857), symptoms (4777) |

## 3. Input vs. Target Expansion Behavior
- **Mean clinical_report length:** `17.5 words (105.9 characters)`
- **Mean summary length:** `25.1 words (162.7 characters)`
- **Expansion Ratio (Target/Input):** `1.54x`
- **Clinical Insight:** In this dataset, summaries are **longer** than the raw input notes because the target expands abbreviations and clinical shorthand into full grammatically complete medical sentences.
