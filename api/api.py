import time
from flask import Flask, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import requests
import flask

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

    def __repr__(self):
        return '<User %r>' % self.email

    @staticmethod
    def create(gid, name, email, profile_pic):
        new_user = User(gid=gid, name=name, email=email, profile_pic=profile_pic)
        db.session.add(new_user)
        db.session.commit()

    @staticmethod
    def get(id):
      return User.query.get(id)

    @staticmethod
    def exists(gid):
      return User.query.filter_by(gid=gid).first() is None

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

class UserLearningPreference(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
  user = db.relationship('User', backref=db.backref('userlearningpreferences', lazy=True))
  question_number = db.Column(db.Integer, nullable=False)
  answer_choice = db.Column(db.Integer, nullable=False)

  def __repr__(self):
        return '<UserLearningPreference %r %r>' % self.user_id, self.question_number

  @staticmethod
  def get_by_user(user_id):
      return User.query.get(id)

  @staticmethod
  def create(user, question_number, answer_choice):
        new_preferences = UserLearningPreference(user=user, question_number=question_number, answer_choice=answer_choice)
        db.session.add(new_preferences)
        db.session.commit()

class Syballus(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  subject = db.Column(db.Text, unique=True, nullable=False)

  def __repr__(self):
        return '<Syballus %r>' % self.subject

  @staticmethod
  def create(subject):
        new_syllabus = Syballus(subject=subject)
        db.session.add(new_syllabus)
        db.session.commit()

class Topic(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  syllabus_id = db.Column(db.Integer, db.ForeignKey('syllabus.id'), nullable=False)
  syllabus = db.relationship('Syllabus', backref=db.backref('topics', lazy=True))
  name = db.Column(db.Text, nullable=False)

  def __repr__(self):
        return '<Topic %r>' % self.name

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

class UserVideoInteraction(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
  user = db.relationship('User', backref=db.backref('uservideointeractions', lazy=True))
  video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
  video = db.relationship('Video', backref=db.backref('uservideointeractions', lazy=True))
  affinity = db.Column(db.Integer, nullable=False)

  def __repr__(self):
      return '<Video %r>' % self.url

  @

  @staticmethod
  def create(user, video, affinity):
        new_video = Video(topic=topic, url=url)
        db.session.add(new_video)
        db.session.commit()
  
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

  response = flask.make_response(info)
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

  return flask.redirect(authorization_url)


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