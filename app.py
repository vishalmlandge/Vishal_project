from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from bson import ObjectId
import bcrypt
import datetime
import re
from datetime import timedelta
import pytz
from toxic_model.detector import score_comment
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/chat_app'  # Update for Render if using external MongoDB
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
mongo = PyMongo(app)
jwt = JWTManager(app)

def validate_email(email):
    email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    return re.match(email_regex, email) is not None

def validate_mobile(mobile):
    mobile_regex = r'^\d{10}$'
    return re.match(mobile_regex, mobile) is not None

def validate_password(password):
    return len(password) >= 6

def is_marathi_hindi(text):
    """
    Check if the text contains Marathi or Hindi characters (Devanagari script).
    """
    devanagari_regex = r'[\u0900-\u097F]'
    return bool(re.search(devanagari_regex, text))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    data = request.get_json()
    email = data.get('email')
    password = data.get('password', '').encode('utf-8')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    user = mongo.db.users.find_one({'email': email})
    if user and bcrypt.checkpw(password, user['password']):
        access_token = create_access_token(identity=str(user['_id']))
        return jsonify({'token': access_token, 'username': user['name']}), 200
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    mobile = data.get('mobile')
    password = data.get('password', '').encode('utf-8')
    confirm_password = data.get('confirm_password', '').encode('utf-8')

    if not all([name, email, mobile, password, confirm_password]):
        return jsonify({'error': 'All fields are required'}), 400
    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400
    if not validate_mobile(mobile):
        return jsonify({'error': 'Mobile number must be 10 digits'}), 400
    if not validate_password(data.get('password')):
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if password != confirm_password:
        return jsonify({'error': 'Passwords do not match'}), 400

    if mongo.db.users.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 400
    if mongo.db.users.find_one({'mobile': mobile}):
        return jsonify({'error': 'Mobile number already exists'}), 400

    hashed = bcrypt.hashpw(password, bcrypt.gensalt())
    user_id = mongo.db.users.insert_one({
        'name': name,
        'email': email,
        'mobile': mobile,
        'password': hashed
    }).inserted_id
    access_token = create_access_token(identity=str(user_id))
    return jsonify({'token': access_token, 'username': name}), 201

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('new_password', '').encode('utf-8')

    if not email or not new_password:
        return jsonify({'error': 'Email and new password are required'}), 400
    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400
    if not validate_password(data.get('new_password')):
        return jsonify({'error': 'New password must be at least 6 characters'}), 400

    user = mongo.db.users.find_one({'email': email})
    if not user:
        return jsonify({'error': 'Email not found'}), 404

    hashed = bcrypt.hashpw(new_password, bcrypt.gensalt())
    mongo.db.users.update_one(
        {'email': email},
        {'$set': {'password': hashed}}
    )
    return jsonify({'message': 'Password reset successfully'}), 200

@app.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    current_user_id = get_jwt_identity()
    return jsonify({'id': current_user_id}), 200

@app.route('/contacts', methods=['GET'])
@jwt_required()
def get_contacts():
    current_user_id = get_jwt_identity()
    users = mongo.db.users.find({'_id': {'$ne': ObjectId(current_user_id)}})
    contacts = [{'id': str(user['_id']), 'username': user['name']} for user in users]
    return jsonify(contacts), 200

@app.route('/chat/<receiver_id>', methods=['GET', 'POST'])
@jwt_required()
def chat(receiver_id):
    try:
        logger.info(f"Received request for /chat/{receiver_id} with method {request.method}")
        logger.info(f"Authorization header: {request.headers.get('Authorization')}")
        
        current_user_id = get_jwt_identity()
        try:
            receiver_id_obj = ObjectId(receiver_id)
        except Exception as e:
            logger.error(f"Invalid receiver ID: {str(e)}")
            return jsonify({'error': 'Invalid receiver ID'}), 400

        if request.method == 'GET':
            query = {
                '$or': [
                    {'sender_id': ObjectId(current_user_id), 'recipient_id': receiver_id_obj},
                    {'sender_id': receiver_id_obj, 'recipient_id': ObjectId(current_user_id)}
                ],
                'deleted_for': {'$ne': ObjectId(current_user_id)}  # Exclude messages deleted for the current user
            }
            logger.info(f"Executing query: {query}")
            messages = mongo.db.messages.find(query).sort('timestamp', 1)
            chat_history = []
            for msg in messages:
                # Use score_comment function from detector.py for toxicity
                toxicity = score_comment(msg['text'])
                logger.info(f"Message text: {msg['text']}, Timestamp: {msg['timestamp']}, Sender: {msg['sender_id']}")
                logger.info(f"Toxicity output: {toxicity}")
                chat_history.append({
                    '_id': str(msg['_id']),  # Include MongoDB _id
                    'sender_id': str(msg['sender_id']),
                    'text': msg['text'],
                    'timestamp': msg['timestamp'].isoformat(),
                    'toxicity': toxicity
                })
            logger.info(f"Returning {len(chat_history)} messages")
            return jsonify(chat_history), 200
        else:
            data = request.get_json()
            text = data.get('text')
            if not text:
                return jsonify({'error': 'Message text is required'}), 400

            # Use score_comment function from detector.py for toxicity
            toxicity = score_comment(text)
            logger.info(f"Sent message text: {text}")
            logger.info(f"Toxicity output: {toxicity}")
            utc_timezone = pytz.timezone('UTC')
            utc_timestamp = datetime.datetime.now(utc_timezone)
            message_doc = {
                'sender_id': ObjectId(current_user_id),
                'recipient_id': receiver_id_obj,
                'text': text,
                'timestamp': utc_timestamp,
                'toxicity': toxicity,
                'deleted_for': []  # Initialize deleted_for as an empty list
            }
            logger.info(f"Inserting message: {message_doc}")
            result = mongo.db.messages.insert_one(message_doc)
            message_doc['_id'] = str(result.inserted_id)  # Include the inserted _id
            return jsonify({
                '_id': str(result.inserted_id),  # Return the _id
                'text': text,
                'toxicity': toxicity,
                'timestamp': utc_timestamp.isoformat()
            }), 200
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat/<receiver_id>/message/<message_id>', methods=['DELETE'])
@jwt_required()
def delete_message(receiver_id, message_id):
    try:
        current_user_id = get_jwt_identity()
        try:
            receiver_id_obj = ObjectId(receiver_id)
            message_id_obj = ObjectId(message_id)  # Convert message_id to ObjectId
        except Exception as e:
            logger.error(f"Invalid ID: {str(e)}")
            return jsonify({'error': 'Invalid ID'}), 400

        # Find the message by _id
        message = mongo.db.messages.find_one({
            '_id': message_id_obj,
            '$or': [
                {'sender_id': ObjectId(current_user_id), 'recipient_id': receiver_id_obj},
                {'sender_id': receiver_id_obj, 'recipient_id': ObjectId(current_user_id)}
            ]
        })

        if not message:
            return jsonify({'error': 'Message not found'}), 404

        # Check if the user is the sender (only sender can delete for everyone)
        is_sender = str(message['sender_id']) == current_user_id
        data = request.get_json()
        delete_option = data.get('option')  # "me" or "everyone"

        if not delete_option or delete_option not in ['me', 'everyone']:
            return jsonify({'error': 'Invalid delete option'}), 400

        if delete_option == 'everyone' and not is_sender:
            return jsonify({'error': 'Only the sender can delete for everyone'}), 403

        if delete_option == 'me':
            # Add current user to deleted_for list
            mongo.db.messages.update_one(
                {'_id': message['_id']},
                {'$addToSet': {'deleted_for': ObjectId(current_user_id)}}
            )
            return jsonify({'message': 'Message deleted for you'}), 200
        else:
            # Delete for everyone
            mongo.db.messages.update_one(
                {'_id': message['_id']},
                {'$addToSet': {'deleted_for': {'$each': [ObjectId(current_user_id), receiver_id_obj]}}}
            )
            return jsonify({'message': 'Message deleted for everyone'}), 200

    except Exception as e:
        logger.error(f"Error in /chat/{receiver_id}/message/{message_id} endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))  # Use Render's PORT or default to 10000
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
