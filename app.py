from flask import Flask, jsonify, request
# from flask_sqlalchemy import SQLAlchemy
# from flask_marshmallow import Marshmallow
from sqlalchemy import null
# import datetime
from eulerian_folder import main1
from moviepy.editor import *
import SIH_trial

app = Flask(__name__)

""" app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)


class Articles(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    body = db.Column(db.Text())
    date = db.Column(db.DateTime, default=datetime.datetime.now)

    def __init__(self, title, body):
        self.title = title
        self.body = body


class ArticleSchema(ma.Schema):
    class Meta:
        fields = ('id', 'title', 'body', 'date')


article_schema = ArticleSchema()
articles_schema = ArticleSchema(many=True)

video_path = "../SIH_backend/uploads/rohin_0001.mov"


@app.route('/get', methods=['GET'])
def get_articles():
    print("hitting")
    all_articles = Articles.query.all()
    results = articles_schema.dump(all_articles)
    return jsonify(results)
    # res = main1.heartrateprediction(video_path)
    # return res


@app.route('/get/<id>/', methods=['GET'])
def post_details(id):
    article = Articles.query.get(id)
    return article_schema.jsonify(article)


@app.route('/add', methods=['POST'])
def add_articles():
    title = request.json['title']
    body = request.json['body']

    articles = Articles(title, body)
    db.session.add(articles)
    db.session.commit()
    return article_schema.jsonify(articles)


@app.route('/update/<id>/', methods=['PUT'])
def update_article(id):
    title = request.json['title']
    body = request.json['body']

    article = Articles.query.get(id)
    article.title = title
    article.body = body
    db.session.commit()
    return article_schema.jsonify(article)


@app.route('/delete/<id>/', methods=['DELETE'])
def delete_article(id):
    article = Articles.query.get(id)
    db.session.delete(article)
    db.session.commit()
    return article_schema.jsonify(article) """


@app.route('/upload', methods=['POST'])
def upload_file():
    print("hitting")
    file = request.files['file']
    # resp = jsonify({'message': 'upload failed'})
    if file:
        file.save(f'uploads/sample.mp4')
        print("video saved")
        # clip = VideoFileClip('uploads/001.mp4')
        # clip.close()
    if VideoFileClip('uploads/sample.mp4') != null:
        print("gettting model...")
        # res = SIH_trial.predictionmodel("uploads/sample.mp4")
        res = main1.heartrateprediction("uploads/sample.mp4")
        # print(res)
        print("giving op...")
    """ return jsonify(
        {'heartrate': res[0],
         'breathingrate': res[1]}
    ) """
    return jsonify({
        'heartrate': res
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
