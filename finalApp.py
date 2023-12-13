import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import string
from heapq import nlargest
from transformers import pipeline
from keybert import KeyBERT
import streamlit as st

def load_nlp_model():
    return spacy.load('en_core_web_sm')

def preprocess_text(text):
    return nlp(text)

def calculate_word_freq(doc):
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq:
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
    return word_freq



def summarize_file(file):
    doc = preprocess_text(file)
    word_freqs = calculate_word_freq(doc)
    sent_tokens = [sent for sent in doc.sents]
    sentencescores = calculate_sentence_scores(sent_tokens, word_freqs)
    select_length = int(len(sent_tokens) * 0.3)
    summary_list = nlargest(select_length, sentencescores, key=sentencescores.get)
    final_summary = [word.text for word in summary_list]
    summary = ' '.join(final_summary)
    return summary



def extract_keywords(text):
    kw_model = KeyBERT()
    key_word = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None)
    return [key[0] for key in key_word]

def main():
    """ EeasyDoc """

    # Title
    st.title("EeasyDoc")

    # Input Text
    input_text = st.text_area("Enter Text", "Type Here ..")

    # Summarization
    if st.checkbox("Show Summary"):
        st.subheader("Summarize Your Text")
        summary_options = st.selectbox("Choose Summarizer", ['orginal summary'])
        if st.button("Summarize"):
            if summary_options == 'orginal summary':
                st.text("Using orginal Summarizer ..")
                output = summarize_file(input_text)
                st.success(output)
                

    # Keyword Extraction
    if st.checkbox("Show Keyword Extraction"):
        st.subheader("Extract Keywords")
        if st.button("Extract"):
            keywords = extract_keywords(input_text)
            st.success(", ".join(keywords))

    st.sidebar.subheader("About App")
    st.sidebar.text("Transforers model using summary, question-answer system, and other tasks in NLP App with Streamlit")

    st.sidebar.subheader("By")
    st.sidebar.text("Tech warrior")


if __name__ == '__main__':
    nlp = load_nlp_model()
    stopwords = list(STOP_WORDS)
    punctuation = punctuation + '\n'
    hugg = pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    main()
