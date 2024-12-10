# Bangla-Hate-Speech-Detection
## Abstract
The rise in abusive language on social media is a significant threat to mental health and social cohesion. For Bengali speakers, the need for effective detection is critical. However, current methods fall short in addressing the massive volume of content. Improved techniques are urgently needed to combat online hate speech in Bengali. Traditional machine learning techniques, while useful, often require large, linguistically diverse datasets to train models effectively. This paper addresses the urgent need for improved hate speech detection methods in Bengali, aiming to fill the existing research gap. Contextual understanding is crucial in differentiating between harmful speech and benign expressions. Large language models (LLMs) have shown state-of-the-art performance in various natural language tasks due to their extensive training on vast amounts of data. We explore the application of LLMs, specifically GPT-3.5 Turbo and Gemini 1.5 Pro, for Bengali hate speech detection using Zero-Shot and Few-Shot Learning approaches. Unlike conventional methods, Zero-Shot Learning identifies hate speech without task-specific training data, making it highly adaptable to new datasets and languages. Few-Shot Learning, on the other hand, requires minimal labeled examples, allowing for efficient model training with limited resources. Our experimental results show that LLMs outperform traditional approaches. In this study, we evaluate GPT-3.5 Turbo and Gemini 1.5 Pro on multiple datasets. To further enhance our study, we consider the distribution of comments in different datasets and the challenge of class imbalance, which can affect model performance. The BD-SHS dataset consists of 35,197 comments in the training set, 7542 in the validation set, and 7542 in the test set. The Bengali Hate Speech Dataset v1.0 and v2.0 include comments distributed across various hate categories: personal hate (629), political hate (1771), religious hate (502), geopolitical hate (1179), and gender abusive hate (316). The Bengali Hate Dataset comprises 7500 non-hate and 7500 hate comments. GPT-3.5 Turbo achieved impressive results with 97.33%, 98.42%, and 98.53% accuracy. In contrast, Gemini 1.5 Pro showed lower performance across all datasets. Specifically, GPT-3.5 Turbo excelled with significantly higher accuracy compared to Gemini 1.5 Pro. These outcomes highlight a 6.28% increase in accuracy compared to traditional methods, which achieved 92.25%. Our research contributes to the growing body of literature on LLM applications in natural language processing, particularly in the context of low-resource languages.

## Table of Contents
- [Citation](#citation)
- [Contact Information](#contact-information)


## Contact Information

For any questions, collaboration opportunities, or further inquiries, please feel free to reach out:

- **Fatema Tuj Johora Faria**
  - Email: [fatema.faria142@gmail.com](mailto:fatema.faria142@gmail.com)

- **Laith H. Baniata**
  - Email: [laith@gachon.ac.kr](mailto:laith@gachon.ac.kr)

- **Sangwoo Kang**
  - Email: [swkang@gachon.ac.kr](mailto:swkang@gachon.ac.kr)

## Citation

If you find the dataset or the associated research work helpful, please consider citing our paper:

```bibtex
@Article{math12233687,
AUTHOR = {Faria, Fatema Tuj Johora and Baniata, Laith H. and Kang, Sangwoo},
TITLE = {Investigating the Predominance of Large Language Models in Low-Resource Bangla Language over Transformer Models for Hate Speech Detection: A Comparative Analysis},
JOURNAL = {Mathematics},
VOLUME = {12},
YEAR = {2024},
NUMBER = {23},
ARTICLE-NUMBER = {3687},
URL = {https://www.mdpi.com/2227-7390/12/23/3687},
ISSN = {2227-7390},
ABSTRACT = {The rise in abusive language on social media is a significant threat to mental health and social cohesion. For Bengali speakers, the need for effective detection is critical. However, current methods fall short in addressing the massive volume of content. Improved techniques are urgently needed to combat online hate speech in Bengali. Traditional machine learning techniques, while useful, often require large, linguistically diverse datasets to train models effectively. This paper addresses the urgent need for improved hate speech detection methods in Bengali, aiming to fill the existing research gap. Contextual understanding is crucial in differentiating between harmful speech and benign expressions. Large language models (LLMs) have shown state-of-the-art performance in various natural language tasks due to their extensive training on vast amounts of data. We explore the application of LLMs, specifically GPT-3.5 Turbo and Gemini 1.5 Pro, for Bengali hate speech detection using Zero-Shot and Few-Shot Learning approaches. Unlike conventional methods, Zero-Shot Learning identifies hate speech without task-specific training data, making it highly adaptable to new datasets and languages. Few-Shot Learning, on the other hand, requires minimal labeled examples, allowing for efficient model training with limited resources. Our experimental results show that LLMs outperform traditional approaches. In this study, we evaluate GPT-3.5 Turbo and Gemini 1.5 Pro on multiple datasets. To further enhance our study, we consider the distribution of comments in different datasets and the challenge of class imbalance, which can affect model performance. The BD-SHS dataset consists of 35,197 comments in the training set, 7542 in the validation set, and 7542 in the test set. The Bengali Hate Speech Dataset v1.0 and v2.0 include comments distributed across various hate categories: personal hate (629), political hate (1771), religious hate (502), geopolitical hate (1179), and gender abusive hate (316). The Bengali Hate Dataset comprises 7500 non-hate and 7500 hate comments. GPT-3.5 Turbo achieved impressive results with 97.33%, 98.42%, and 98.53% accuracy. In contrast, Gemini 1.5 Pro showed lower performance across all datasets. Specifically, GPT-3.5 Turbo excelled with significantly higher accuracy compared to Gemini 1.5 Pro. These outcomes highlight a 6.28% increase in accuracy compared to traditional methods, which achieved 92.25%. Our research contributes to the growing body of literature on LLM applications in natural language processing, particularly in the context of low-resource languages.},
DOI = {10.3390/math12233687}
}

