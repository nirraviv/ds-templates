from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
from pycontractions import Contractions
import gensim.downloader as api

nlp = spacy.load('en_core_web_lg')  # or 'en_core_web_md'

# cont = Contractions(api_key='gensim_model')
# cont = Contractions('disk_model')
model = api.load('glove-twitter-25')
cont = Contractions(kv_model=model)
cont.load_models()


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, 'html.parser')
    stripped_text = soup.get_text(separator=' ')
    return stripped_text


def remove_extra_whitespace(text):
    """remove extra whitespace from text"""
    text = text.strip()  # remove leading and trailing whitespace
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café -> cafe"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text, **kwargs):
    """expand shortened words, e.g. don't -> do not"""
    # choose model
    text = list(cont.expand_texts([text], precise=True))[0]
    return text


def text_preprocessing(text, accented_chars=True, contractions=True, convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, stop_words=True, **kwargs):
    # TODO: add regex preprocesssing
    if remove_html:
        text = strip_html_tags(text)
    if extra_whitespace:
        text = remove_extra_whitespace(text)
    if accented_chars:
        text = remove_accented_chars(text)
    if contractions:
        text = expand_contractions(text, **kwargs)
    if 'exclude_stop_words' in kwargs.keys():
        for word in list(kwargs['exclude_stop_words']):
            nlp.vocab[word].is_stop = False
    if 'customize_stop_words' in kwargs.keys():
        for word in list(kwargs['customize_stop_words']):
            nlp.vocab[word].is_stop = True
    if lowercase:
        text = text.lower()

    doc = nlp(text)

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations and token.pos_ == 'PUNCT' and flag:
            flag = False
        # remove special characters
        if special_chars and token.pos_ == 'SYM' and flag:
            flag = False
        # remove numbers
        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False
        # convert number words to number numbers
        if convert_num and token.pos_ == 'NUM' and flag:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization and token.lemma_ != "-PRON-" and flag:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag:
            clean_text.append(edit)
    return clean_text


def test():
    text = "I'd like to have three cups   of coffee<br /><br />from your Café. #delicious"
    clean_text = text_preprocessing(text)
    print(clean_text)


if __name__ == '__main__':
    test()
