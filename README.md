# Tanglish Tweets Sentiment Analysis using XLM-Roberta 

## Overview

For the past week, I’ve randomly developed a *fascination* with **Sentiment Analysis** (don’t ask why, even I don’t know 😅). But once I started diving into it, I was hooked. The way machines attempt to interpret human emotions — through messy, inconsistent, and deeply personal *language* — absolutely amazed me.

So here I am. But instead of using yet another generic, overused dataset that everyone trains on blindly without even relating to it, I wanted to do something **different**, something **authentic**.

###  What is Sentiment Analysis?

Sentiment Analysis is a subfield of NLP (Natural Language Processing) where the goal is to **analyze textual data** and classify the *emotional tone* behind it — usually as **positive**, **negative**, or **neutral**. It tries to capture what people *feel*, not just what they say. This makes it incredibly powerful for social media analysis, brand monitoring, or just understanding raw human emotion in text.

##  The Dataset

While exploring datasets, I stumbled upon this little gem on Kaggle:

🔗 [Tanglish Comments for Sentiment Analysis](https://www.kaggle.com/datasets/vyombhatia/tanglish-comments-for-sentiment-ananlysis)

> The dataset contains **Tanglish** comments — that's Tamil + English code-mixed language — mostly scraped from social media. It reflects real, local expressions of sentiment in a bilingual setting. This makes the task more *challenging* and *fun* because regular English-based sentiment models can’t deal well with such mixed language text.

This dataset was perfect for what I had in mind. It’s quirky, raw, and reflects real-life, relatable emotions from a region I connect with.

##  The Model — XLM-Roberta

To handle the *code-mixed multilingual nature* of Tanglish, I fine-tuned the **XLM-Roberta (Base)** model. It’s a powerful transformer model pre-trained on 100+ languages, making it well-suited for tasks involving Indian languages + English.

### Why XLM-Roberta?

-  Multilingual understanding
-  Large-scale pretraining (on CommonCrawl + Wikipedia)
-  Performs well on zero-shot and fine-tuned settings
-  Great for underrepresented languages

##  What I Did

-  Preprocessed the Tanglish dataset
-  Tokenized using `XLMRobertaTokenizer`
-  Fine-tuned `XLMRobertaForSequenceClassification` with HuggingFace's `Trainer` API
-  Tracked and visualized training progress using **Weights & Biases (WandB)**

##  Training Pipeline

1. Load & inspect dataset
2. Tokenize using `xlm-roberta-base`
3. Format into PyTorch tensors
4. Train on sentiment labels (positive/negative)
5. Evaluate and log performance metrics

##  Results

- Achieved promising accuracy (details coming soon 😉)
- Model was able to adapt well to Tanglish despite limited training data
- Planning further experiments on emoji-handling & sarcasm detection 👀

## 📚 Dependencies

```bash
pip install transformers datasets evaluate sentencepiece wandb
