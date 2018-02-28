from datetime import datetime
import os
import stat

from flask import Flask, json, jsonify, request, make_response
from healthcheck import HealthCheck
from jsonschema import validate, ValidationError
import pandas as pd
from sklearn.externals import joblib

from model import train
from schemas.payload_schema import payload_schema


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['MAIN_MODEL'] = os.environ.get('CREDITOR_MAIN_MODEL')
app.config['MAIN_MODEL_ID'] = app.config['MAIN_MODEL'][:-4]
app.config['MODELS_FOLDER'] = os.path.join(BASE_DIR, 'models')
app.config['DATAFILES_FOLDER'] = os.path.join(BASE_DIR, 'datafiles')
app.config['ALLOWED_EXTENSIONS'] = {'parquet'}
app.config['VERSION_CONTROL'] = os.path.join(app.config['MODELS_FOLDER'], '.versions')

# Example request: curl -X GET http://localhost/healthcheck
health = HealthCheck(app, "/healthcheck")


# Healthcheck section

def models_available():

    """Checks the availability of the main model and if it is predicting"""

    model_filepath = os.path.join(app.config['MODELS_FOLDER'], app.config['MAIN_MODEL'])
    model = joblib.load(model_filepath)
    sample_data = {
        'score_3': 420.0,
        'score_4': 99.384867,
        'score_5': 0.661019,
        'score_6': 106.788234
    }
    sample_input = pd.DataFrame.from_dict([sample_data], orient='columns')
    _ = model.predict_proba(
        sample_input[[
            'score_3',
            'score_4',
            'score_5',
            'score_6'
        ]])[:, 1][0]
    return True, "Model loading and prediction are ok"


health.add_check(models_available)


# Error handling section

# Class InvalidUsage taken from Flask documentation
class InvalidUsage(Exception):
    # Defaults to Bad Request
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'message': 'Resource not found'}), 404)


# Helper functions section

def allowed_file(filename):

    """Checks if the file has an allowed file extension"""

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def exists(model_id):

    """Checks if model <model_id> is available in the disk"""

    return '{}.pkl'.format(model_id) in os.listdir(app.config['MODELS_FOLDER'])


def validate_json_payload(payload):

    """Validates input data against a predefined JSON schema"""

    try:
        validate(payload, payload_schema)
    except ValidationError:
        raise InvalidUsage('JSON schema mismatch: please review your input data')


def update_versions(current_version):

    """ Adds current version to the version control file """

    versions_file = os.path.join(app.config['MODELS_FOLDER'], '.versions')
    with open(versions_file, 'r') as f:
        try:
            models_versions = json.load(f)
        except ValueError:
            models_versions = []
    models_versions.append(current_version)
    with open(versions_file, 'w') as f:
        json.dump(models_versions, f)


# Views section

@app.route("/predict", methods=['POST'])
@app.route("/predict/<string:model_id>", methods=['POST'])
def predict(model_id=app.config['MAIN_MODEL_ID']):

    """Predicts default risk for a potential client

    POST: Predicts default risk for a potential client using the Main Logistic
    Regression Model (endpoint /predict) or using the Logistic Regression model defined by <model_id>
    (endpoint /predict/<model_id>); body is a JSON content payload with input data. Returns HTTP 200 on success;
    Returns HTTP 400 in case of JSON schema mismatch or if the model defined by <model_id> does not exist

    Example requests:

    curl -H "Content-Type: application/json" -X POST -d <json_input> http://localhost/predict
    curl -H "Content-Type: application/json" -X POST -d <json_input> http://localhost/predict/<model_id>
    """

    if not exists(model_id):
        raise InvalidUsage(
            'Model {} does not exist. Check the running versions at /versions'.format(model_id),
            status_code=404
        )
    else:
        data = json.loads(request.data)
        validate_json_payload(data)
        input_data = pd.DataFrame.from_dict([data], orient='columns')
        model_filepath = os.path.join(app.config['MODELS_FOLDER'], '{}.pkl'.format(model_id))
        model = joblib.load(model_filepath)
        output_data = {
            'id': data['id'],
            'probability': model.predict_proba(input_data[['s1', 's2', 's3', 's4']])[:, 1][0]
        }
        return jsonify(output_data)


@app.route("/update/<string:model_id>", methods=['POST'])
def update(model_id):

    """Updates Logistic Regression model when more data are available

    POST: Updates Logistic Regression model when more data are available; body is a
    binary .parquet file hosting the data. Returns HTTP 200 on success; Returns HTTP 400
    if the model defined by <model_id> already exists or HTTP 413 if the file size is
    greater than 250 MB (can be redefined by the web server configuration)

    Example request:
    curl -H "Content-Type: multipart/form-data" -F "file=@/path/to/file.parquet" http://localhost/update/<model_id>
    """

    if exists(model_id):
        raise InvalidUsage('Model {} already exists. Please choose another name'.format(model_id))
    else:
        datafile = request.files['file']
        if datafile and allowed_file(datafile.filename):
            data_filepath = os.path.join(app.config['DATAFILES_FOLDER'], '{}.parquet'.format(model_id))
            datafile.save(data_filepath)
            model, auc = train(data_filepath)
            model_filepath = os.path.join(app.config['MODELS_FOLDER'], '{}.pkl'.format(model_id))
            joblib.dump(model, model_filepath)

            model_creation_time = datetime.fromtimestamp(os.stat(model_filepath)[stat.ST_CTIME])
            current_version = {'id': model_id, 'created_at': str(model_creation_time), 'auc': auc}
            update_versions(current_version)

            return jsonify({'message': 'Your new model {} is up and running'.format(model_id)})
        else:
            raise InvalidUsage('File not found or file extension not allowed')


@app.route("/versions", methods=['GET'])
def versions():

    """Shows the list of running models

    GET: Shows the list of running models with <model_id> and creation data for reference.
    Returns HTTP 200 on success

    Example request:
    curl -X GET http://localhost/versions
    """

    versions_file = os.path.join(app.config['MODELS_FOLDER'], '.versions')
    return jsonify(json.loads(open(versions_file, 'r').read()))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
