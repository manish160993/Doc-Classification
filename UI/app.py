# Import required modules

import flask
import pickle
from flask import Flask, request
from static.document_classifier import classify_input

# Create an instance of Flask and declare variables for our model
application = Flask(__name__)
clf = pickle.load(open('./static/saved_data_objects/LRModel.pkl', 'rb'))
vectorizer = pickle.load(open('./static/saved_data_objects/vectorizer.pkl','rb'))
id_to_category = pickle.load(open('./static/saved_data_objects/id_to_category.pkl','rb'))
# app.run(debug=True,port=12345)


# Homepage, render input prompt HTML
@application.route('/')
def index():
    return flask.render_template('index.html')
    
# Result page, get prediction result and render
@application.route('/predict', methods=['GET', 'POST'])
# Handle request
def form():
    # Document content was sent with POST request
    if request.method == 'POST':
        # Preparation
        data = request.form['input_data']
        result = classify_input(data, clf, vectorizer, id_to_category)
        return flask.render_template('result.html', document = data, result=result)

if __name__ == '__main__':
    application.run()