# AITextDetection
This work investigates various methodologies, including Support Vector Machines (SVM) with RBF kernels, Bidirectional Long Short-Term Memory (BiLSTM) networks with attention mechanisms, and the Mistral model, for classifying machine generated text. We evaluate these models on the M4 dataset and MGTBench for out-of-distribution data, as well as on mixed data within MixSet. Our results indicate that SVMs equipped with RBF kernels and BiLSTMs augmented with attention mechanisms significantly outperform other models in their respective categories with over **96%** accuracy over M4 dataset and **97%** accuracy on unseen GPT4 generated data. Additionally, we find that the choice of data mixing method crucially impacts the effectiveness of the detectors, with the humanization approach posing the greatest challenge.

# Document
Please read the following document to know the detail behind this project:
- [Proposal](https://github.com/derek33125/AITextDetection/blob/main/Document/Project%20Proposal%20-%20GPT%20refined%20version.pdf)
- [MReport](https://github.com/derek33125/AITextDetection/blob/main/Document/Project%20MileStone%20-%20GPT%20refined%20Version.pdf)
- [Final Report](https://github.com/derek33125/AITextDetection/blob/main/Document/Aist4010_Final_Report_GPT%20refined%20Version.pdf)
- [Presentation](https://github.com/derek33125/AITextDetection/blob/main/Document/1155158302_Derek_Presentation.pptx)

# Table Of Contents
-  [Getting Started](#getting-started)
-  [In Details](#in-details)
    -  [Project architecture](#project-architecture)
       -  [SVM](#SVM)
       -  [BiLSTM series](#BiLSTM-Series)
       -  [Mistral 7B](#Mistral-7B)
    -  [Folder structure](#folder-structure)

# Getting Started   
- In data generation, you can modify the code inside the **Data_Generation** folder to alter the training and testing dataset based on you need. If not, you can directly use the generated data inside the **Code** folder for direct testing.
- In the **Code** folder, it already provides all the models' training and inference code for getting the results. Notice that the evaluation metrics in MixSet is a bit different from other testing dataset so you may need to uncomment or modify the testing method when doing evaluation.
- For Mistral7B, please ensure that you are having at least A100 GPU before training and testing.
**Notice that the final accuracy in MixSet should be 1 - original accuracy due to the objective in testing that dataset. Detail is listed in the code**


# In Details

Project architecture 
--------------

### SVM
To task the performance of those traditional machine-learning method for classification of LLM generated text with human text, I incorporated the SVM for text classification. Before deploying the model, the preprocessed the data will further employs the TFIDF to converts the text data into a matrix of TFIDF features to enabling the SVM to know the textual features. The TFIDF vectors is set with the maximum length of 1000. After that, the radius basis function will be act as the kernel of SVM and do the classification. I tried evaluating its performance using different kernels (Linear, Ploy, Sigmoid, RBF).

### BiLSTM Series

<div align="center">
<img align="center" hight="600" width="600" src="https://github.com/derek33125/AITextDetection/blob/main/Figures_Results_Graphs/Figures/BiLSTM/BiLSTM%20series%20models.png?raw=true">

</div>

### Mistral 7B
The Mistral 7B model is another key component of this study. According to the model’s foundationalpaper, Mistral 7B significantly outperforms the popular Llama 2 13B across all benchmarks, and evensurpasses Llama 34B on many benchmarks. This demonstrates that smaller language models canmatch the capabilities of larger ones when optimized correctly. For this project,we utilize the Mistral-7B variant enhanced with Block-wise Model-update Filtering and Bit-centering(BNB), which boosts model efficiency and reduces memory demands. Additionally, we employ a quantized 4-bit version of the model Face, facilitating training on T4 GPUs by minimizing the model’s size.In the implementation phase, the ‘FastLanguageModel‘ from the UnSLoth library AI is used to download Mistral-7B and set the maximum sequence length to 2048 tokens. Furthermore, LoRA technology is applied to train only 4% of the model’s parameters, utilizing techniques such as gradient accumulation and precision training to enhance training efficiency. Unlike the standard natural language processing approaches used with the SVM and LSTM models, this phase involves Supervised Fine-Tuning. Here, text data and their corresponding labels are formatted into prompts suitable for retraining the model on the machine-text classification task. Training is conducted using the PEFT technique Face combined with the SFT Trainer Face , optimizing the
model’s performance in text classification.

Folder structure
--------------

```
├──  Code
│   ├── aist4010_project_mistral7b.ipynb  - the running code for mistral7b
│   ├── aist4010_cnn-lstm.ipynb   - the running code for BiLSTM series models.
|   └── aist4010_svm.ipynb  - the running code for SVM series model.
│
│
├── Data_Generation - this folder contains the original data source
│   ├── M4
|   ├── MGTBench
|   ├── MixSet
|   └── Project Code.ipynb - the code for generating the datasets
│
│
├── Document  - this folder contains all the written documents
│
|   
├── Figures_Results_Graphs - this folder contains all the generated results
|   ├── Data - this folder contains the photos regarding the selected datasets
|   ├── Figures - this folder contains the generated figure results
|   ├── result - this folder contains the numeric results
|   └── Training Log - this folder contains the training log for each model
│
| 
└── Plotting Graphs - this folder cotains all the result for plotting the graph
    ├── LSTM_general.json
    ├── LSTM_MGTBench.json
    ├── LSTM_Mix.json
    ├── SVM_general.json
    ├── SVM_MGTBench.json
    ├── SVM_Mix.json
    └── Plotting Results.ipynb

```


## Configuration
Even though I wrote the methods for saving the testing results of different testing datasets into JSON files seperately, I did not write any code for further filtering the needed data to plot the figures automatically. You may either do it manually or writing method yourself. For some models like SVM, you may need to reduce the training dataset size first since it is very time consuming in training process.
