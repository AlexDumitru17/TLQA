# TimeLine based List Question Answering (TLQA)

<p align="center">
  <img src="TLQA.png" />
</p>

# Data Collection pipeline
<p align="center">
  <img src="pipeline_tlqa.png" />
</p>

##  :file_folder: File Structure

```
.
├── LICENSE.MD
├── README.md
├── TLQA.png
├── TLQA_data_splits
│   ├── benchmark_v0.0.json
│   └── splits
│       ├── add_wikipedia_titles.py
│       ├── get_golden_evidence.py
│       ├── test_split_benchmark_v0.0.json
│       ├── test_split_benchmark_v0.0_golden_evidence.json
│       ├── test_split_benchmark_v0.0_updated_with_titles.json
│       ├── train_split_benchmark_v0.0.json
│       ├── train_split_benchmark_v0.0_golden_evidence.json
│       └── train_split_benchmark_v0.0_updated_with_titles.json
├── TempLama
│   ├── data
│   │   ├── templama_test.json
│   │   ├── templama_train.json
│   │   └── templama_val.json
│   ├── evaluateData.py
│   ├── evaluateResults.py
│   ├── experiment.py
│   ├── gpt3_responses.csv
│   ├── listqas.json
│   ├── output.json
│   ├── queryGPT.py
│   ├── tempLama.py
│   ├── templama_test.json
│   ├── templama_train.json
│   └── templama_val.json
├── data_pipeline
│   ├── offices_parser.mjs
│   ├── sport_parser.mjs
│   ├── tempLama.py
│   └── test.py
├── evaluation
│   └── evaluateResults.py
├── output.txt
├── test.py
├── tlqa_1.png
└── tlqa_2.png
```
