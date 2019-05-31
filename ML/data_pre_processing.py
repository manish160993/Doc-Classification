'''File to pre-process data and save the processed an transformed data for further analysis'''
import sys
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

def preprocess_basic(data):
    '''remove null values, convert text category to ids'''
    # drop null values
    data_non_null = data.dropna()
    # convert text category into ids
    data_non_null['Category_id'] = data_non_null['Category'].factorize()[0]
    return data_non_null

def label_to_id(data):
    '''dictionary of id to category and vice-versa'''
    category_id_df = data[['Category','Category_id']].drop_duplicates().sort_values('Category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['Category_id','Category']].values)
    # save dictionary
    pickle.dump(id_to_category, open('saved_data_objects/id_to_category.pkl', 'wb'))
    return category_to_id, id_to_category

def split_data(data):
    '''split data intro train, test and validation'''
    corpus = data.drop(['Category'], axis=1)
    features = corpus['Document']
    label = corpus['Category_id']
    train_feature, test_feature, train_label, test_label = \
        train_test_split(features, label, test_size=0.3, random_state = random_seed)
    x_train, x_val, y_train, y_val = train_test_split(train_feature, train_label,
                                                  test_size = .1,
                                                  random_state= random_seed)
    return x_train, x_val, y_train, y_val, test_feature, test_label

def feature_extraction(x_train, x_val, test_feature):
    '''convert training text of documents to tf-idf features'''
    vectorizer = TfidfVectorizer(min_df=0.2)
    x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
    x_val_tfidf = vectorizer.transform(x_val).toarray()
    test_feature_tfidf = vectorizer.transform(test_feature).toarray()
    # save vectorizer
    pickle.dump(vectorizer, open('saved_data_objects/vectorizer.pkl', 'wb'))
    return x_train_tfidf, x_val_tfidf, test_feature_tfidf

def data_upsampling(x_train_tfidf, y_train):
    ''' data upsampling since classes are unbalanced '''
    sm = SMOTE('minority',random_state=random_seed)
    x_train_res, y_train_res = sm.fit_sample(x_train_tfidf, y_train)
    num_categories = len(y_train.value_counts())
    for i in range(num_categories):
        x_train_res, y_train_res = sm.fit_sample(x_train_res, y_train_res)
    return x_train_res, y_train_res

def save_data_files(x_train_res,y_train_res,x_val_tfidf,y_val,test_feature_tfidf,test_label):
    '''save data files as csv'''
    pd.DataFrame(test_feature_tfidf).to_csv('data/x_test.csv', index=False)
    test_label.to_csv('data/y_test.csv', index=False)
    pd.DataFrame(x_train_res).to_csv('data/x_train.csv',index=False)
    pd.DataFrame(x_val_tfidf).to_csv('data/x_val.csv', index=False)
    pd.DataFrame(y_train_res).to_csv('data/y_train.csv',index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    

if __name__=="__main__":
    random_seed = 0
    dataset_location = sys.argv[1]
    data = pd.read_csv(dataset_location, sep=",", names=["Category","Document"])
    os.makedirs("saved_data_objects")
    data = preprocess_basic(data)
    category_to_id, id_to_category = label_to_id(data)
    x_train, x_val, y_train, y_val, test_feature, test_label = split_data(data)
    x_train_tfidf, x_val_tfidf, test_feature_tfidf = feature_extraction(x_train, x_val, test_feature)
    x_train_res, y_train_res = data_upsampling(x_train_tfidf, y_train)
    os.makedirs("data")
    save_data_files(x_train_res,y_train_res,x_val_tfidf,y_val,test_feature_tfidf,test_label)
    print("Data Pre Processing and Transformation finished")
    

  

