import time
from flask import Flask, redirect, session, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import numpy as np
import requests
import flask
import sys

sys.path.insert(1, '../recommendation')

from inference import inference

import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://ryan:{os.environ["POSTGRES_PASS"]}@localhost/curio'
db = SQLAlchemy(app)
CORS(app)


os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gid = db.Column(db.Text, unique=True, nullable=False)
    name = db.Column(db.Text, unique=True, nullable=False)
    email = db.Column(db.Text, unique=True, nullable=False)
    profile_pic = db.Column(db.Text, unique=False, nullable=False)
    
    most_recent_syllabus_id = db.Column(db.Integer, db.ForeignKey('syllabus.id'), nullable=True)
    most_recent_syllabus = db.relationship('Syllabus', backref=db.backref('users', lazy=True))
    most_recent_topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), nullable=True)
    most_recent_topic = db.relationship('Topic', backref=db.backref('users', lazy=True))

    def __repr__(self):
        return '<User %r>' % self.email

    def to_dict(self):
      syllabus = None
      if self.most_recent_syllabus is not None:
        syllabus = self.most_recent_syllabus.to_dict()

      topic = None
      if self.most_recent_topic is not None:
        topic = self.most_recent_topic.to_dict()

      return {'id': self.id,
      'name': self.name,
      'email': self.email,
      'profile_pic': self.profile_pic,
      'most_recent_syllabus': syllabus,
      'most_recent_topic': topic
      }

    @staticmethod
    def create(gid, name, email, profile_pic):
        new_user = User(gid=gid, name=name, email=email, profile_pic=profile_pic, most_recent_syllabus=None, most_recent_topic=None)
        db.session.add(new_user)
        db.session.commit()

    @staticmethod
    def get(id):
      return User.query.get(id)

    @staticmethod
    def get_by_gid(gid):
      return User.query.filter_by(gid=gid).first()

    @staticmethod
    def exists(gid):
      user = User.query.filter_by(gid=gid).first()
      return user is not None

class UserLearningPreference(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
  user = db.relationship('User', backref=db.backref('userlearningpreferences', lazy=True))
  question_number = db.Column(db.Integer, nullable=False)
  answer_choice = db.Column(db.Integer, nullable=False)

  def __repr__(self):
        return f'<UserLearningPreference {self.user_id} {self.question_number}>'

  def to_dict(self):
    return {
      'id': self.id,
      'user': self.user.to_dict(),
      'question_number': self.question_number,
      'answer_choice': self.answer_choice
    }

  @staticmethod
  def get_by_user(user_id):
      return UserLearningPreference.query.filter_by(user_id=user_id).order_by(UserLearningPreference.question_number).all()

  @staticmethod
  def create(user, question_number, answer_choice):
        new_preferences = UserLearningPreference(user=user, question_number=question_number, answer_choice=answer_choice)
        db.session.add(new_preferences)
        db.session.commit()

class Syllabus(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  subject = db.Column(db.Text, unique=True, nullable=False)

  def __repr__(self):
        return '<Syllabus %r>' % self.subject

  def to_dict(self):
    return {
      'id': self.id,
      'subject': self.subject
    }

  @staticmethod
  def create(subject):
        new_syllabus = Syllabus(subject=subject)
        db.session.add(new_syllabus)
        db.session.commit()

  @staticmethod
  def get(id):
    return Syllabus.query.get(id)

class Topic(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  syllabus_id = db.Column(db.Integer, db.ForeignKey('syllabus.id'), nullable=False)
  syllabus = db.relationship('Syllabus', backref=db.backref('topics', lazy=True))
  name = db.Column(db.Text, nullable=False)

  def __repr__(self):
        return '<Topic %r>' % self.name

  def to_dict(self):
    return {
      'id': self.id,
      'syllabus': self.syllabus.to_dict(),
      'name': self.name
    }

  @staticmethod
  def create(syllabus, name):
        new_topic = Topic(syllabus=syllabus, name=name)
        db.session.add(new_topic)
        db.session.commit()

class Video(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), nullable=False)
  topic = db.relationship('Topic', backref=db.backref('videos', lazy=True))
  url = db.Column(db.Text, unique=True, nullable=False)

  def __repr__(self):
      return '<Video %r>' % self.url

  @staticmethod
  def create(topic, url):
        new_video = Video(topic=topic, url=url)
        db.session.add(new_video)
        db.session.commit()
    
  @staticmethod
  def get_by_url(url):
    return Video.query.filter_by(url=url).first()

class UserVideoInteraction(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
  user = db.relationship('User', backref=db.backref('uservideointeractions', lazy=True))
  video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
  video = db.relationship('Video', backref=db.backref('uservideointeractions', lazy=True))
  affinity = db.Column(db.Integer, nullable=False)

  def __repr__(self):
      return '<Video %r>' % self.url

  @staticmethod
  def create(user, video, affinity):
        new_interaction = UserVideoInteraction(user=user, video=video, affinity=affinity)
        db.session.add(new_interaction)
        db.session.commit()

@app.route('/interaction', methods=['POST'])
def mark_interaction():
  if 'credentials' not in flask.session:
    response = flask.make_response({})
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response
  
  print(request.form['affinity'])

  credentials = google.oauth2.credentials.Credentials(
      **flask.session['credentials'])
  client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)

  info = client.userinfo().get().execute()

  current_user = User.get_by_gid(info['id'])

  vid = Video.get_by_url(request.form['video'])

  UserVideoInteraction.create(current_user, vid, float(request.form['affinity']))

  response = flask.make_response('Success!')
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response

@app.route('/video/<int:topic_id>')
def get_video(topic_id):
    if 'credentials' not in flask.session:
      response = flask.make_response({})
      response.headers['Access-Control-Allow-Credentials'] = 'true'
      return response

    credentials = google.oauth2.credentials.Credentials(
      **flask.session['credentials'])
    client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)
    info = client.userinfo().get().execute()

    current_user = User.get_by_gid(info['id'])
    current_user.most_recent_topic = Topic.query.get(topic_id)
    db.session.commit()

    prefs = UserLearningPreference.get_by_user(current_user.id)

    converted = []

    for pref in prefs:
      if pref.answer_choice == 0:
        converted += [1, 0, 0]
      elif pref.answer_choice == 1:
        converted += [0, 1, 0]
      else:
        converted += [0, 0, 1]

    interactions = UserVideoInteraction.query.filter_by(user_id=current_user.id).all()

    affinities = []
    videos = []

    for i in interactions:
      affinities.append(i.affinity)
      videos.append(Video.query.get(i.video_id).url)

    possible_videos = Video.query.filter_by(topic=current_user.most_recent_topic).all()

    best_id = inference(np.array(converted), np.array(affinities), videos, possible_videos, 0)

    response = flask.make_response({'id' : best_id})
    response.headers['Access-Control-Allow-Credentials'] = 'true'

    return response
    


@app.route('/topic/<int:syllabus_id>')
def get_topics(syllabus_id):
  result = Topic.query.filter_by(syllabus_id=syllabus_id).all()

  response = flask.make_response(jsonify([i.to_dict() for i in result]))
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response

@app.route('/syllabus')
def get_syllabi():
  result = Syllabus.query.all()

  response = flask.make_response(jsonify([i.to_dict() for i in result]))
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response

@app.route('/surveyrespond', methods=['POST'])
def set_survey_response():
  if 'credentials' not in flask.session:
    response = flask.make_response('You must be logged in!')
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response
  
  if len(request.form) != 20:
    return 'Invalid survey, not all items were entered.'
  

  credentials = google.oauth2.credentials.Credentials(
      **flask.session['credentials'])
  client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)

  info = client.userinfo().get().execute()

  current_user = User.get_by_gid(info['id'])

  for question, answer in request.form.items():
    print(question)
    print(answer)
    UserLearningPreference.create(current_user, int(question[1:]), int(answer))

  response = flask.make_response(redirect('http://localhost:3000/'))
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response


@app.route('/surveyresponses')
def get_survey_responses():
  if 'credentials' not in flask.session:
    response = flask.make_response({})
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

  credentials = google.oauth2.credentials.Credentials(
      **flask.session['credentials'])
  client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)

  info = client.userinfo().get().execute()

  current_user = User.get_by_gid(info['id'])

  survey_responses = UserLearningPreference.query.filter_by(user=current_user).order_by(UserLearningPreference.question_number).all()

  response = flask.make_response(jsonify([i.to_dict() for i in survey_responses]))
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response
  

# Authentication Routes 
CLIENT_SECRETS_FILE = "../client_secret.json"

SCOPES = ['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile', 'openid']

app.secret_key = os.environ["CURIO_SECRET_KEY"]

@app.route('/userinfo')
def get_user_info():
  if 'credentials' not in flask.session:
    response = flask.make_response({})
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

  # Load credentials from the session.
  credentials = google.oauth2.credentials.Credentials(
      **flask.session['credentials'])

  client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)

  flask.session['credentials'] = credentials_to_dict(credentials)

  info = client.userinfo().get().execute()

  user = User.get_by_gid(info['id'])

  response = flask.make_response(user.to_dict())
  response.headers['Access-Control-Allow-Credentials'] = 'true'

  return response


@app.route('/login')
def authorize():
  # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
  flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
      CLIENT_SECRETS_FILE, scopes=SCOPES)

  # The URI created here must exactly match one of the authorized redirect URIs
  # for the OAuth 2.0 client, which you configured in the API Console. If this
  # value doesn't match an authorized URI, you will get a 'redirect_uri_mismatch'
  # error.
  flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

  authorization_url, state = flow.authorization_url(
      # Enable offline access so that you can refresh an access token without
      # re-prompting the user for permission. Recommended for web server apps.
      access_type='offline',
      # Enable incremental authorization. Recommended as a best practice.
      include_granted_scopes='true')

  # Store the state so the callback can verify the auth server response.
  flask.session.permanent = True
  flask.session['state'] = state

  return redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
  # Specify the state when creating the flow in the callback so that it can
  # verified in the authorization server response.
  state = flask.session['state']

  flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
      CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
  flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

  # Use the authorization server's response to fetch the OAuth 2.0 tokens.
  authorization_response = flask.request.url
  flow.fetch_token(authorization_response=authorization_response)

  credentials = flow.credentials
  flask.session['credentials'] = credentials_to_dict(credentials)

  client = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)

  info = client.userinfo().get().execute()

  if not User.exists(info['id']):
    User.create(info['id'], info['name'], info['email'], info['picture'])

  return redirect('http://localhost:3000/')


@app.route('/revoke')
def revoke():
  if 'credentials' not in flask.session:
    return ('You need to <a href="/login">login</a> before ' +
            'testing the code to revoke credentials.')

  credentials = google.oauth2.credentials.Credentials(
    **flask.session['credentials'])

  revoke = requests.post('https://oauth2.googleapis.com/revoke',
      params={'token': credentials.token},
      headers = {'content-type': 'application/x-www-form-urlencoded'})

  status_code = getattr(revoke, 'status_code')
  if status_code == 200:
    return('Credentials successfully revoked.')
  else:
    return('An error occurred.')


@app.route('/clear')
def clear_credentials():
  if 'credentials' in flask.session:
    del flask.session['credentials']
  return ('Credentials have been cleared.')


def credentials_to_dict(credentials):
  return {'token': credentials.token,
          'refresh_token': credentials.refresh_token,
          'token_uri': credentials.token_uri,
          'client_id': credentials.client_id,
          'client_secret': credentials.client_secret,
          'scopes': credentials.scopes}