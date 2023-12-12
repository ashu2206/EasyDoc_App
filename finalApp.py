import streamlit as st
import pdfplumber

def pdftotxt(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open("sample.txt", "a") as f:
            for i, page in enumerate(pdf.pages):
                file = page.extract_text()
                f.write(file)

    with open("sample.txt", "r") as f:
        text = f.read()

    return text

def simple_summary(text, sentence_count=5):
    sentences = text.split('.')  # Split text into sentences
    selected_sentences = sentences[:sentence_count]
    summary_result = '. '.join(selected_sentences)
    return summary_result

def main():
    """ EasyDoc """

    # Title
    st.title("EasyDoc")

    st.markdown("""
      #### Description
        + This is a Natural Language Processing(NLP) Based App useful for basic NLP tasks
        Summarization
        """)

    # Upload PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Summarization
    if uploaded_file is not None:
        st.subheader("Summarize Your Text")

        # Read PDF content
        pdf_text = pdftotxt(uploaded_file)

        message = st.text_area("Enter Text", pdf_text)
        summary_options = st.selectbox("Choose Summarizer", ['original summary'])

        if st.button("Summarize"):
            if summary_options == 'original summary':
                st.text("Using original Summarizer ..")
                summary_result = simple_summary(message)
                st.success(summary_result)
            else:
                st.warning("Invalid summarizer option selected.")

    st.sidebar.subheader("About App")
    st.sidebar.text("NLPiffy App with Streamlit")
    st.sidebar.info("Kudos to the Streamlit Team")

    st.sidebar.subheader("By")
    st.sidebar.text("Anwesha Roy")
    st.sidebar.text("Ashu Baghel")
    st.sidebar.text("Harshika Saxena")
    st.sidebar.text("Srishti Saxena")

if __name__ == '__main__':
    main()