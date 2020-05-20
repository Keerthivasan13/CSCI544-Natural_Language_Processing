# CSCI544-Natural_Language_Processing  
 Code developed for CSCI544 Course taught by Prof. Mark Core.  

**Language Used -** Python 3.7  

## 1. Spam Filtering using Naive Bayes Classifier ##  

**Basic Functionalities :** nbLearn, nbClassify, nbEvaluate  

**Accuracy :** Spam F1 Score - 98%, Ham F1 Score - 95%

**Enhanced Functionalities :** modification.py, m_classify.py  
 1. Replacing numbers with a default unique token - "NUMBER"  
 2. Added Stopword filter with common stopwords from NLTK corpus  
 3. Added Stopword filter with handpicked tokens such as "Subject:", ":", "\", "the", "and"...  

 **Accuracy :** Spam F1 Score - 99%, Ham F1 Score - 97%
 
 **For more details please refer Report.txt**
 
 ## 2. Sequence Labeling using Conditional Random Fields ##
 
 **Library :** pycrfsuite  
 **Dataset :** The Switchboard Corpus (SWBD) Dialog Tags Annotations (DAMSL). More => https://web.stanford.edu/~jurafsky/ws97/manual.august1.html  
 
 **Baseline Features :**
 1) Speaker changed from previuous Utterance
 2) First Utterance
 3) Token in an Utterance
 4) Part of Speechtag in an Utterance
 
 **Accuracy :** 62 %
 
 **Advanced Features :**
 1) Last Utterance in a Dialogue
 2) First Token in an Utterance
 3) Last Token in an Utterance
 4) First POS in an Utterance
 5) Last POS in an Utterance
 6) Bigrams of Tokens
 7) Bigrams of POS
 8) Bigram of last token in the previous utterance and first token in the current utterance
 9) Individual word in the text field (Text Column in the CSV with Noise)
 
  **Accuracy :** 67 %
  
  **For more details please refer Report.txt**
