from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)  # 다른 포트번호에 대한 보안 제거

    from .views import main_views
    app.register_blueprint(main_views.bp)

    return app
