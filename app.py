import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def find_best_match(user_question, questions):
    sentence_embeddings = []
    for sentence in [user_question] + questions:
        tokens = nlp(sentence)
        sentence_vector = np.mean([token.vector for token in tokens], axis=0)
        sentence_embeddings.append(sentence_vector)
    
    # Calculate cosine similarity between the specific sentence and others
    similarity_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
    
    # Find the index of the most similar sentence
    most_similar_index = np.argmax(similarity_scores)
    
    # Get the most similar sentence
    closest_sentence = questions[most_similar_index]
    
    return closest_sentence
    


def get_answer(user_question, selected_image, data):
    print(user_question)
    if user_question == "":
        return "Please enter a question"
    for image_data in data['images']:
        if image_data['image'] == selected_image:
            questions = [item['question'] for item in image_data['questions']]
            best_match = find_best_match(user_question, questions)
            if best_match:
                for item in image_data['questions']:
                    if item['question'] == best_match:
                        return item['answer']

    return "Please select and image first"

@app.route('/api/answer', methods=['POST'])
def answer_question():
    user_question = request.json.get('question')
    selected_image = request.json.get('selectedImage')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data.json')
    data = load_data(file_path)
    print(selected_image, "selected_image")
    answer = get_answer(user_question, selected_image, data)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
