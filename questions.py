import nltk
import sys
import os
import string
import math
nltk.download('punkt')
nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = {}
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as f:
            result[file] = f.read()

    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document.lower())

    result = [
        word for word in words if word not in string.punctuation and
        word not in nltk.corpus.stopwords.words("english")
    ]

    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_result = {}
    len_doc = len(documents)

    unique = set(sum(documents.values(), []))

    for word in unique:
        count = 0
        for document in documents.values():
            if word in document:
                count += 1

        idf_result[word] = math.log(len_doc/count)

    return idf_result


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}
    for fname, fcontent in files.items():
        fscore = 0
        # Calculate for each word tf-idf value for each file
        for word in query:
            if word in fcontent:
                fscore += fcontent.count(word) * idfs[word]
        if fscore != 0:
            tf_idf[fname] = fscore

    tf_idf_sort = [
        key for key, value in
        sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    ]

    return tf_idf_sort[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank = {}
    for sentence, words in sentences.items():
        idf = 0
        for word in query:
            if word in words:
                # Update idfs
                idf += idfs[word]

        if idf != 0:
            # Update query term density
            density = sum([words.count(word) for word in query]) / len(words)
            rank[sentence] = (idf, density)

    idf_sort = [
        key for key, value in
        sorted(rank.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    ]

    return idf_sort[:n]


if __name__ == "__main__":
    main()
