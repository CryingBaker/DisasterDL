import os
from flask import Flask
from flask_cors import CORS
from routes.bd_dataset import bd_dataset_bp
from routes.bd_predict import bd_predict_bp
from routes.fs_dataset import fs_dataset_bp
from routes.fs_predict import fs_predict_bp
from routes.intelligence import intelligence_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(bd_dataset_bp, url_prefix='/api/bd_dataset')
app.register_blueprint(bd_predict_bp, url_prefix='/api/bd_predict')
app.register_blueprint(fs_dataset_bp, url_prefix='/api/fs_dataset')
app.register_blueprint(fs_predict_bp, url_prefix='/api/fs_predict')
app.register_blueprint(intelligence_bp, url_prefix='/api/intelligence')

if __name__ == '__main__':
    # NOTE: debug=False prevents Werkzeug's auto-reloader from spawning a
    # watcher process that blocks startup while models are loading.
    app.run(host='0.0.0.0', port=5000, debug=False)
