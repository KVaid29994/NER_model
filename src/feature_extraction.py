from src.logger_config import get_logger

logger = get_logger(__name__)

def word2features(sent, i):
    word = str(sent[i][0])
    features = {
        'word': word,
        'is_first': i == 0,
        'lowercase': word.lower(),
        'is_title': word.istitle(),
        'is_upper': word.isupper(),
        'is_digit': word.isdigit(),
        'is_alphanumeric': word.isalnum(),
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'has_hyphen': '-' in word,
        'prev_word': '' if i == 0 else sent[i-1][0],
        'next_word': '' if i == len(sent)-1 else sent[i+1][0],
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:lower': word1.lower(),
            '-1:is_title': word1.istitle()
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:lower': word1.lower(),
            '+1:is_title': word1.istitle()
        })
    else:
        features['EOS'] = True
    return features

def prepare_data(sentences):
    logger.info("Preparing features and labels from sentences")
    X = [[word2features(s, i) for i in range(len(s))] for s in sentences]
    y = [[tag for (_, tag) in s] for s in sentences]
    return X, y

def prepare_single_sentence(sentence: str):
    logger.info("Preparing features from single input sentence")
    tokens = sentence.split()
    wrapped = [(w,) for w in tokens]  
    return [word2features(wrapped, i) for i in range(len(wrapped))]
