from flask import Flask, render_template, request
from app import views

app = Flask(__name__, template_folder="app/static/templates", static_folder="app/static")
app.register_blueprint(views.views, url_prefix="/")


if __name__ == '__main__':
    app.run(debug=True, port=8000)
