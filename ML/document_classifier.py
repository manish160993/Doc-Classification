'''File to test unknown data against saved ML model'''
import sys
import pickle 

def classify_input(inp, model, vectorizer, id_to_category):
    '''function to classify an input word or string into a document according to our model'''
    input_vector = [inp]
    transformed_input = vectorizer.transform(input_vector).toarray()
    predicted_category_id = model.predict(transformed_input)
    predicted_doc_category = id_to_category[predicted_category_id[0]]
    return predicted_doc_category

if __name__=="__main__":
    input_text = sys.argv[1]
    clf = pickle.load(open('saved_data_objects/LRModel.pkl', 'rb'))
    vectorizer = pickle.load(open('saved_data_objects/vectorizer.pkl','rb'))
    id_to_category = pickle.load(open('saved_data_objects/id_to_category.pkl','rb'))
    classify_input(input_text, clf, vectorizer, id_to_category)