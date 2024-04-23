import streamlit as st 
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context 
 
from PyPDF2 import PdfReader 
from transformers import pipeline 
from keybert import KeyBERT 
import spacy 
import base64  # Add base64 module for encoding 
from spacy.lang.en.stop_words import STOP_WORDS 
from string import punctuation 
from heapq import nlargest 
 
 
# Add the provided functions for text summarization 
def load_nlp_model(): 
    return spacy.load('en_core_web_sm') 
 
def preprocess_text(text, nlp): 
    return nlp(text) 
 
def calculate_word_freq(doc): 
    word_freq = {} 
    for word in doc: 
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation: 
            if word.text not in word_freq: 
                word_freq[word.text] = 1 
            else: 
                word_freq[word.text] += 1 
    return word_freq 
 
def calculate_sentence_scores(sent_tokens, word_freqs): 
    sent_scores = {} 
    for sent in sent_tokens: 
        for word in sent: 
            if word.text.lower() in word_freqs: 
                if sent not in sent_scores: 
                    sent_scores[sent] = word_freqs[word.text.lower()] 
                else: 
                    sent_scores[sent] += word_freqs[word.text.lower()] 
    return sent_scores 
 
def summarize_text(text, nlp): 
    doc = preprocess_text(text, nlp) 
    word_freqs = calculate_word_freq(doc) 
    sent_tokens = [sent for sent in doc.sents] 
    sentencescores = calculate_sentence_scores(sent_tokens, word_freqs) 
    select_length = int(len(sent_tokens) * 0.3) 
    summary_list = nlargest(select_length, sentencescores, key=sentencescores.get) 
    final_summary = [word.text for word in summary_list] 
    summary = ' '.join(final_summary) 
    return summary 
 
# Function to extract text from PDF file 
def extract_text_from_pdf(file): 
    pdf_reader = PdfReader(file) 
    text = "" 
    for page in pdf_reader.pages: 
        text += page.extract_text() 
    return text 
 
# Function for question answering 
def question_answer(text, question): 
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2") 
    result = qa_pipeline(question=question, context=text) 
    return result["answer"] 
 
# Function for keyword extraction 
def extract_keywords(text): 
    kw_model = KeyBERT() 
    keywords = kw_model.extract_keywords(text) 
    return [key[0] for key in keywords] 
 
def main(): 
    """ EeasyDoc """ 
 
    # Title 
    st.title("EeasyDoc") 
 
    # File upload 
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"]) 
 
    if uploaded_file is not None: 
        file_text = extract_text_from_pdf(uploaded_file) 
 
        # Summarization 
        if st.checkbox("Show Summary"): 
            st.subheader("Summarize Your Text") 
            if st.button("Summarize"): 
                # Load Spacy NLP model 
                nlp = load_nlp_model() 
                summary = summarize_text(file_text, nlp) 
                st.success(summary) 
 
        # Question-answer 
        if st.checkbox("Show Question Answer system"): 
            st.subheader("Get Your answer") 
            question = st.text_input("Enter Question") 
            if st.button("Answer"): 
                answer = question_answer(file_text, question) 
                st.success(answer) 
 
        # Keyword extraction 
        if st.checkbox("Show Keywords"): 
            st.subheader("Extracted Keywords") 
            keywords = extract_keywords(file_text) 
            st.success(keywords) 
 
        # Optional download button for the summary in text format 
        if st.button("Download Summary as Text"): 
            with st.spinner("Downloading..."): 
                # Load Spacy NLP model 
                nlp = load_nlp_model() 
                summary = summarize_text(file_text, nlp) 
                # Encode summary text in base64 
                summary_encoded = base64.b64encode(summary.encode()).decode() 
                href =f"data:file/txt;base64,{summary_encoded}" 
                st.markdown(f"📥 Download Summary as Text", unsafe_allow_html=True) 
 
    st.sidebar.subheader("About App") 
    st.sidebar.text("Transformers model using summarization, question-answer system, and other tasks in NLP App with Streamlit") 
 
    st.sidebar.subheader("By") 
    st.sidebar.text("Tech warrior") 
 
if __name__ == '__main__': 
    main()
