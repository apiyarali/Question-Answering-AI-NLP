# Question Answering AI
This project implements a simple question answering system using natural language processing (NLP) techniques, specifically focusing on document and passage retrieval based on a user's query. The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to rank documents and sentences in terms of their relevance to the query.

## Overview
The AI answers questions based on a corpus of text files (Wikipedia articles in this case). The process is divided into two main tasks:

1. **Document Retrieval**: The system finds the most relevant documents to a given query.

2. **Passage Retrieval**: After identifying the relevant documents, the system extracts the most relevant sentences or passages from those documents.

The system ranks documents using a combination of term frequency (TF) and inverse document frequency (IDF). Once the top documents are identified, sentences within those documents are scored based on how well they match the query, again using IDF and a query term density measure.

## Getting Started
### Prerequisites
- Python 3.x
- Install the necessary dependencies using:

    ```bash
    pip3 install -r requirements.txt
    ```

- `nltk` (Natural Language Toolkit) is required for text processing and tokenization.

### Files and Directories
- **corpus/**: A directory containing `.txt` files that represent the documents from which the AI will extract answers.
- **questions.py**: The main script that implements the question answering AI.
- **requirements.txt**: A file listing the dependencies for this project.


### How to Run
To interact with the AI, run the following command in the terminal:

```bash
python questions.py corpus
```
The program will prompt you to enter a query, and it will then return the most relevant sentence from the corpus based on your input.

Example queries:

1. **Query:** "What are the types of supervised learning?"

- **Answer:** "Types of supervised learning algorithms include Active learning, classification, and regression."

2. **Query:**  "When was Python 3.0 released?"

- **Answer:** "Python 3.0 was released on 3 December 2008."

3. **Query:** "How do neurons connect in a neural network?"

- **Answer:** "Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers."

## Implementation Details
### Functions
- **load_files(directory):** Loads the text files from the specified directory and returns a dictionary mapping filenames to file contents.

- **tokenize(document):** Tokenizes a given document into a list of lowercase words, filtering out punctuation and stopwords.

- **compute_idfs(documents):** Computes the Inverse Document Frequency (IDF) for each word in the documents. This helps in ranking documents and sentences based on their relevance to the query.

- **top_files(query, files, idfs):** Given a query and a set of files, this function ranks the files by their relevance to the query using TF-IDF.

- **top_sentences(query, sentences, idfs):** Given a query and sentences from the top files, this function ranks the sentences based on their relevance to the query using IDF and query term density.

### Example Workflow
- The user runs the AI with the corpus directory.
- The system loads and tokenizes the documents in the corpus.
- It computes IDF values for all words in the corpus.
- The user enters a query, and the system uses the top_files function to find the most relevant document(s).
- It then uses the top_sentences function to find the most relevant sentence(s) from the identified documents.
- The AI returns the top matching sentence as the answer.

## Customization
- **Corpus Modification:** You can modify the corpus directory by adding or removing .txt files to experiment with different datasets.
- **Global Settings:** The number of files and sentences to retrieve for each query can be adjusted by modifying the FILE_MATCHES and SENTENCES_MATCHES variables in questions.py.

## Notes
- This implementation focuses on simplicity and performance using TF-IDF and basic NLP techniques.
- More sophisticated techniques, such as semantic analysis, lemmatization, or advanced ranking algorithms, can be implemented as future improvements.

## Conclusion
This project demonstrates a basic yet functional question answering system using TF-IDF for document and passage retrieval. With further enhancements, it can be expanded to handle more complex queries and diverse datasets.

## License
This project is part of an Harvard's CS80 AI coursework assignment exploring Natural Language Processing (NLP)

