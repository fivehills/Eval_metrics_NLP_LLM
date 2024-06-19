

# About
This project is to overview **Evaluation Metrics for NLP tasks and LLMs performance** and propose new effective metrics based on the significance of statistical analysis.


# Evaluation Metrics for NLP Tasks

## Basic Metrics 

For evaluating NLP tasks, the following metrics are often employed.


**Exact match (EM):** The percentage of predictions that match any one of the answers exactly.

**(Macro-averaged) F1 score (F1):** Each answer and prediction is tokenized into words. For every answer to a given question, the overlap between the prediction and each answer is calculated and the maximum F1 is chosen. This score is then averaged over all the questions. Formally speaking:

F1 = (2 * precision * recall) / (precision + recall)  
precision = (number of same tokens) / length(predicted tokens)  
recall = (number of same tokens) / length(labeled tokens)

**Perplexity:** Perplexity is a measurement of how well a probability model predicts a sample. A low perplexity indicates the probability distribution is good at predicting the sample. In NLP, perplexity is a way of evaluating language models. A model of an unknown probability distribution p, may be proposed based on a training sample that was drawn from p. Given a proposed probability model q, one may evaluate q by asking how well it predicts a separate test sample x1, x2, ..., xN also drawn from p. The perplexity of the model q is defined as:

b^(-1/N * sum(i=1 to N) log_b q(xi))

where b is customarily 2. (Martinc, Pollak, and Robnik-Šikonja 2019)

**BLEU:** BLEU (**B**i**l**ingual **E**valuation **U**nderstudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account. (Papineni et al. 2002)

**Accuracy:** Accuracy is the ratio of number of correct predictions to the total number of input samples.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Matthews correlation coefficient**: The MCC is used as a measure of quality of binary classifications. It takes true and false positives and negatives into account and is regarded as a balanced measure which can be used even if the classes are imbalanced. The MCC can be calculated directly from the confusion matrix using the following formula:

MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


### Note: Average precision
   * *Macro*: average of sentence scores
   * *Micro*: corpus (sums numerators and denominators for each hypothesis-reference(s) pairs before division)

## Metrics in Machine Translation
1. **BLEU** (BiLingual Evaluation Understudy)
    * [Papineni 2002](https://www.aclweb.org/anthology/P02-1040.pdf)
    * 'Measures how many words overlap in a given translation when compared to a reference translation, giving higher 
     scores to sequential words.' (recall)
    * Limitation:
        * Doesn't consider different types of errors (insertions, substitutions, synonyms, paraphrase, stems)
        * Designed to be a corpus measure, so it has undesirable properties when used for single sentences.
2. **GLEU** (*Google-BLEU*)
    * [Wu et al. 2016](http://arxiv.org/pdf/1609.08144v2.pdf)
    * Minimum of BLEU recall and precision applied to 1, 2, 3 and 4grams
        * Recall: (number of matching n-grams) / (number of total n-grams in the target)
        * Precision: (number of matching n-grams) / (number of total n-grams in generated sequence)
    * Correlates well with BLEU metric on a corpus metric but does not have its drawbacks for per sentence reward objective.
    * Not to be confused with Generalized Language Evaluation Understanding or *Generalized BLEU*, also known as **GLEU** 
        * Napoles et al. 2015's ACL paper: [*Ground Truth for Grammatical Error Correction Metrics*](http://www.aclweb.org/anthology/P15-2097)
        * Napoles et al. 2016: [*GLEU Without Tuning*](https://arxiv.org/abs/1605.02592)
            * Minor adjustment required as the number of references increases.
        * Simple variant of BLEU, it hews much more closely to human judgements.
        * "In MT, an untranslated word or phrase is almost always an error, but in GEC, this is not the case."
            * GLEU: "computes n-gram precisions over the reference but assigns more weight to n-grams that have been correctly changed from the source." 
        * [Python code](https://github.com/cnap/gec-ranking/)        
3. **WER** (Word Error Rate)
    * Levenshtein distance (edit distance) for words: minimum number of edits (insertion, deletions or substitutions) required to change the hypotheses sentence into the reference.
    * Range: greater than 0 (ref = hyp), no max range as Automatic Speech Recognizer (ASR) can insert an arbitrary number of words
    * $ WER = \frac{S+D+I}{N} = \frac{S+D+I}{S+D+C} $
        * S: number of substitutions, D: number of deletions, I: number of insertions, C: number of the corrects,
            N: number of words in the reference ($N=S+D+C$)
    * WAcc (Word Accuracy) or Word Recognition Rate (WRR): $1 - WER$
    * Limitation: provides no details on the nature of translation errors
        * Different errors are treated equally, even thought they might influence the outcome differently (being more disruptive or more difficult/easier to be corrected).
        * If you look at the formula, there's no distinction between a substitution error and a deletion followed by an insertion error.
    * Possible solution proposed by Hunt (1990):
        * Use of a weighted measure
        * $ WER = \frac{S+0.5D+0.5I}{N} $
        * Problem:
            * Metric is used to compare systems, so it's unclear whether Hunt's formula could be used to assess the performance of a single system
            * How effective this measure is in helping a user with error correction
    * See [more info](https://martin-thoma.com/word-error-rate-calculation/)
4. **METEOR** (Metric for Evaluation of Translation with Explicit ORdering):
    * Banerjee 2005's paper: [*Meteor: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments*](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf)
    * About: "based on the harmonic mean of unigram precision and recall (weighted higher than precision)"
    * Includes: exact word, stem and synonym matching
    * Designed to fix some of the problems found in the BLEU metric, while also producing good correlation with human
        judgement at the sentence or segment level (unlike BLEU which seeks correlation at the corpus level).
    * [Python jar wrapper](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap/meteor)
5. **TER** (Translation Edit Rate)
    * Snover et al. 2006's paper: [A study of translation edit rate with targeted human annotation](https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf)
    * Number of edits (words deletion, addition and substitution) required to make a machine translation match
        exactly to the closest reference translation in fluency and semantics
    * TER = $\frac{E}{R}$ = (minimum number of edits) / (average length of reference text)
    * It is generally preferred to BLEU for estimation of sentence post-editing effort. [Source](http://opennmt.net/OpenNMT/tools/scorer/).
    * [PyTER](https://pypi.python.org/pypi/pyter/0.2.2.1)
    * **char-TER**: character level TER




## The Main Evaluation Datasets

| Name | Task | Size | Description |
| --- | --- | --- | --- |
| SQuAD 2.0 | Question Answering, Reading Comprehension | 150,000 | Paragraphs w questions and answers |
| CoQA | Question Answering, Reading Comprehension | 127,000 | Answering interconnected questions |
| GLUE | General Language Understanding | - | Nine different NLU tasks |
| SuperGLUE | General Language Understanding | - | Eight different NLU tasks |
| AQuA-RAT | Question Answering, Reading Comprehension, Mathematical Reasoning | 100,000 | Solving algebraic word problems |
| SNLI | Natural Language Inference | 570,000 | Understanding entailment and contradiction |
| Irony Sarcasm Analysis Corpus | Classification, Sentiment Analysis | 33,000 | Ironic, sarcastic, regular and figurative tweets |
| WikiText-103 & 2 | Language Modelling | 100M+ | Word and character level tokens from Wikipedia |
| WMT 14 English-German | Language Translation | 4.5M | Sentence pairs for translation |
| VOiCES | Speech Recognition | 3,900 | Voices in complex environmental settings. 15h material |


## Evaluation Dataset Colab Implementations

<table align="center"> 
  <tr>
    <td align="center"><b>Metric</b></td>
    <td align="center"><b>Application</b></td>
    <td align="center" colspan="2"><b>Notebook</b></td>
  </tr>
  <tr>
    <td align="center">BLEU</td><td align="center">Machine Translation</td>
    <td align="center"><a href="https://github.com/gcunhase/NLPMetrics/blob/master/notebooks/bleu.ipynb">Jupyter</a></td>
    <td align="center"><a href="https://colab.research.google.com/github/gcunhase/NLPMetrics/blob/master/notebooks/bleu.ipynb">Colab</a></td>
  </tr>
  <tr>
    <td align="center">GLEU (Google-BLEU)</td><td align="center">Machine Translation</td>
    <td align="center"><a href="https://github.com/gcunhase/NLPMetrics/blob/master/notebooks/gleu.ipynb">Jupyter</a></td>
    <td align="center"><a href="https://colab.research.google.com/github/gcunhase/NLPMetrics/blob/master/notebooks/gleu.ipynb">Colab</a></td>
  </tr>
  <tr>
    <td align="center">WER (Word Error Rate)</td><td align="center">Transcription Accuracy<br>Machine Translation</td>
    <td align="center"><a href="https://github.com/gcunhase/NLPMetrics/blob/master/notebooks/wer.ipynb">Jupyter</a></td>
    <td align="center"><a href="https://colab.research.google.com/github/gcunhase/NLPMetrics/blob/master/notebooks/wer.ipynb">Colab</a></td>
  </tr>
</table>



#🔥 Evaluation Metrics for LLMs performance


- Large variety of ready-to-use LLM evaluation metrics (all with explanations) powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
  - G-Eval
  - Summarization
  - Answer Relevancy
  - Faithfulness
  - Contextual Recall
  - Contextual Precision
  - RAGAS
  - Hallucination
  - Toxicity
  - Bias
  - etc. 
- Evaluate your entire dataset in bulk in under 20 lines of Python code **in parallel**. Do this via the CLI in a Pytest-like manner, or through our `evaluate()` function.
- Create your own custom metrics that are automatically integrated with DeepEval's ecosystem by inheriting DeepEval's base metric class.
- Integrates seamlessly with **ANY** CI/CD environment.
- Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://docs.confident-ai.com/docs/benchmarks-introduction), which includes:
  - MMLU
  - HellaSwag
  - DROP
  - BIG-Bench Hard
  - TruthfulQA
  - HumanEval
  - GSM8K


# NLP eval implementations
## Requirements
Tested on Python 2.7
```
pip install -r requirements.txt
```

## How to Use
* Run: `python test/test_mt_text_score.py`
* Currently only supporting MT metrics
