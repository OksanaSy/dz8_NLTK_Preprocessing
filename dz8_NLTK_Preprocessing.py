import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True,
                 to_lowercase: bool = True, remove_punctuation: bool = True,
                 remove_numbers: bool = True, custom_stopwords: Optional[List[str]] = None):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.to_lowercase = to_lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers

        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> List[str]:
        if self.to_lowercase:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]

        if self.remove_numbers:
            tokens = [token for token in tokens if not token.isdigit()]

        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens


text = "This is an example sentence to demonstrate text preprocessing using NLTK."

preprocessor = TextPreprocessor()
processed_text = preprocessor.preprocess(text)
print(processed_text)
