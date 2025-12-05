from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Import and register blueprints
    from .routes import api
    from .routes_3d import api_3d
    
    app.register_blueprint(api)
    app.register_blueprint(api_3d)

    return app 