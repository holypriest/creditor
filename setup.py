from datetime import datetime
import os
import stat

from sklearn.externals import joblib

from creditor import update_versions
from model import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_MODEL = os.environ.get('CREDITOR_MAIN_MODEL')
MAIN_MODEL_ID = MAIN_MODEL[:-4]
MAIN_DATAFILE = os.environ.get('CREDITOR_MAIN_DATAFILE')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
DATAFILES_FOLDER = os.path.join(BASE_DIR, 'datafiles')
VERSIONS_FILE = os.path.join(MODELS_FOLDER, '.versions')


def on_starting(server):

    """Generates main model and adds it to the version control, if there are no other models"""

    if not open(VERSIONS_FILE, 'r').read():
        initial_model, auc = train(os.path.join(DATAFILES_FOLDER, MAIN_DATAFILE))
        initial_model_filepath = os.path.join(MODELS_FOLDER, MAIN_MODEL)
        joblib.dump(initial_model, initial_model_filepath)

        model_creation_time = datetime.fromtimestamp(os.stat(initial_model_filepath)[stat.ST_CTIME])
        current_version = {'id': MAIN_MODEL_ID, 'created_at': str(model_creation_time), 'auc': auc}
        update_versions(current_version)
