import argparse
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def run_search(args, tokenizer_name, n_tokens, estimators, parameters):
    print(args.filename)
    print(args.text_column)
    print(args.class_column)
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset_name = str(args.filename)
    dataset_name = dataset_name.split("/")[-1].split('.')[0]    
    dataset = pd.read_csv(args.filename)
    text_column = args.text_column
    class_column = args.class_column
    le = LabelEncoder()
    
    X_train, X_test, y_train, y_test = train_test_split(dataset[text_column], dataset[class_column], test_size=0.2)
    
    le.fit(y_train)
    tokenizer = base_tokenizer.train_new_from_iterator(X_train, n_tokens)
    tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize)
    tfidf.fit(X_train)
    
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    joblib.dump(X_train, f"../dados/atv3/X_train_{dataset_name}.pkl")
    joblib.dump(y_train, f"../dados/atv3/y_train_{dataset_name}.pkl")
    joblib.dump(X_test, f"../dados/atv3/X_test_{dataset_name}.pkl")
    joblib.dump(y_test, f"../dados/atv3/y_test_{dataset_name}.pkl")
    
    result = []
    df = pd.DataFrame()
    
    for estimator in estimators:
        gs = GridSearchCV(estimators[estimator], parameters[estimator], n_jobs=4, cv=10, scoring="accuracy")
        gs.fit(X_train, y_train)
        
        aux = pd.DataFrame(gs.cv_results_)
        aux["estimator"] = estimator
        aux["n_tokens"] = n_tokens
        aux["tokenizer"] = tokenizer_name
        aux["best_estimator"] = gs.best_estimator_
        aux["best_parameters"] = str(gs.best_params_)
        df = pd.concat([df, aux])
    
    df.to_csv(f"../dados/atv3/results_gs_{dataset_name}.csv", index=False)
        
if __name__ == "__main__":
    estimators = {'MultinomialNB': MultinomialNB(), 'LogisticRegression': LogisticRegression(), 'KNeighborsClassifier': KNeighborsClassifier()}
    parameters = {'MultinomialNB': {'alpha': [0.1, 0.5, 1.0, 1.5], 'fit_prior': [True, False]}, 'LogisticRegression': {"solver": ["liblinear", "saga"],"penalty": ["l1", "l2"], "C": [0.1, 0.5, 1.0], "fit_intercept": [True, False]}, 'KNeighborsClassifier': {"n_neighbors": [2, 5, 10], "weights": ["uniform", "distance"], "p": [1, 2]}}
    tokenizer_name = "roberta-base"
    n_tokens = 300
    parser = argparse.ArgumentParser(
                    prog='BestHypeparameters',
                    description='Busca de hyperparametros do algoritmo',
                    epilog='Text at the bottom of help')
    parser.add_argument('filename')
    parser.add_argument('text_column')
    parser.add_argument('class_column')
    args = parser.parse_args()
    run_search(args, tokenizer_name, n_tokens, estimators, parameters)