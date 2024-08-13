import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline, PipelineException
import streamlit as st

try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
except PipelineException as e:
    st.error(f"Error loading classification pipeline: {e}")
    st.stop()

def post_analyser(sentence):
    try:
        post_sentiment = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", return_all_scores=True, device=0)
        result = post_sentiment(sentence)
        max_entry = max(result[0], key=lambda x: x['score'])
        max_label = max_entry['label']
        max_score = max_entry['score']
        return max_label, max_score
    except Exception as e:
        st.error(f"Error during post analysis: {e}")
        return None, None

def map_labels(output):
    label_mapping = {
        'LABEL_0': 'positive',
        'LABEL_1': 'negative',
        'LABEL_2': 'neutral'
    }
    
    try:
        mapped_output = [{'label': label_mapping[item['label']], 'score': item['score']} for item in output]
        max_entry = max(mapped_output, key=lambda x: x['score'])
        return max_entry['label'], max_entry['score']
    except Exception as e:
        st.error(f"Error mapping labels: {e}")
        return None, None

def review_analyser(sentence):
    try:
        review_sentiment = pipeline("text-classification", model="Dmyadav2001/Sentimental-Analysis", return_all_scores=True, device=0)
        result = review_sentiment(sentence)
        label, score = map_labels(result[0])
        return label, score
    except Exception as e:
        st.error(f"Error during review analysis: {e}")
        return None, None

st.title("Text Classification and Sentiment Analysis")

input_seq = st.text_input("Enter the text to classify:", "camera can be improved a bit")

candidate_labels = ['review', 'post']

if st.button("Classify and Analyze"):
    try:
        domain = classifier(input_seq, candidate_labels)
        scores = domain['scores']
        labels = domain['labels']
        max_index = scores.index(max(scores))
        label = labels[max_index]

        st.write(f"Classification Label: {label}")

        if label == 'post':
            st.write("Using the model RoBERTa - large - English to analyze the post sentiment")
            post_label, post_score = post_analyser(input_seq)
            if post_label:
                st.write(f"Sentiment: {post_label} (Score: {post_score:.2f})")
        else:
            st.write("Using the model distillBERT to analyze the review sentiment")
            review_label, review_score = review_analyser(input_seq)
            if review_label:
                st.write(f"Sentiment: {review_label} (Score: {review_score:.2f})")
    except Exception as e:
        st.error(f"Error during classification: {e}")
