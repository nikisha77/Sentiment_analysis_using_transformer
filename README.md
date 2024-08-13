
# Sentiment Analysis using Transformers

![image](https://github.com/user-attachments/assets/bf026ec7-2b06-402c-90ee-69a88abdf008)


The process of determining whether a particular sentence convey a postive / negative or a neural emotion is called as sentiment Analysis. This can be used to understand the customer feedbasck , public opinions and etc.

Transformers work by processing text in a way that captures both the meaning of individual words and their relationships within a sentence. This allows them to grasp the insights present in  language, such as sarcasm, irony, and context, which traditional methods often struggle with.

I have made this project to choice the suitable transformer as per the needs (Domain /class which they belong to)

* **Zero-Shot Classification** - Used  to classify text into categories without specific training. here used for the purpose of defining the domain to which the sentence belong to .

* **DistilBERT** -  Leaner version of BERT, offering 97% of its accuracy while being faster and lighter, perfect for real-time sentiment analysis 

* **RoBERTa** -  Enhances BERT by using improved training techniques, making it more robust and effective for analyzing complex language data.



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`HUGGINGFACE_API_KEY`='ENTER YOUR API KEY '
