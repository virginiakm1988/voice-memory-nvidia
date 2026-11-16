# Voice Memory with NVIDIA NIM and NeMo — 繁中快速入門

以 NIM 固定校正模型，透過訓練資料提出文字記憶修改，只有獨立 selection set 的 WER 嚴格改善才接受。測試集不參與記憶更新。另提供 NeMo Speech 音訊輸入與 AutoModel LoRA 對照。

1. 安裝 Python 3.12，執行 `pip install -e .`。
2. 先用 `python -m project_demo demo --mode fixture` 熟悉輸入與評分。
3. 實際資料載入需 `pip install -e ".[cpu,tutorial]"`。
4. 真實模型需對應 NIM／HF 權限及 GPU，使用環境變數或 Colab Secrets 保存憑證。
5. 從 README 開啟 Colab，使用唯一的執行模式選單，讀取保存的結果與錯誤。

- [Interactive quickstart](https://colab.research.google.com/github/virginiakm1988/voice-memory-nvidia/blob/main/notebooks/01_quickstart.ipynb)
- [Evaluation and error analysis](https://colab.research.google.com/github/virginiakm1988/voice-memory-nvidia/blob/main/notebooks/02_evaluation.ipynb)
- [NeMo AutoModel LoRA](https://colab.research.google.com/github/virginiakm1988/voice-memory-nvidia/blob/main/notebooks/03_automodel_lora.ipynb)

## 原始研究與差異

- [Voice Memory paper](https://arxiv.org/abs/2607.26410)
- [Upstream method, code and memory artifacts](https://huggingface.co/huckiyang/voice-memory)
- [HyPoradise corpus](https://huggingface.co/datasets/PeacefulData/HyPoradise-v0)
- [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech)

本 repo 是小規模研究教學。真實模型結果、人工撰寫的離線案例、未完成的 GPU 驗證會明確區分；所有結果保存在獨立執行資料夾。完整方法、授權及限制請閱讀 [English README](README.md)。

實跑結果已加入英文 README 與 evidence：包含真實 NVIDIA SDK 執行、失敗紀錄、LoRA 梯度更新、adapter 儲存重載，以及固定資料前後比較。Notebook 預設離線模式，GPU 訓練須勾選；SpeechEQ 的 ASR 訓練是獨立教學，不代表情緒智慧微調成果。
