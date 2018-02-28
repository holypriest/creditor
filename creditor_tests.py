from datetime import datetime
import os
import stat
import unittest

from flask import json
from sklearn.externals import joblib

from creditor import app, update_versions
from model import train


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['MAIN_MODEL'] = 'askr.pkl'
app.config['MAIN_MODEL_ID'] = app.config['MAIN_MODEL'][:-4]
app.config['MODELS_FOLDER'] = os.path.join(BASE_DIR, 'tests', 'models')
app.config['DATAFILES_FOLDER'] = os.path.join(BASE_DIR, 'tests', 'datafiles')
app.config['VERSION_CONTROL'] = os.path.join(app.config['MODELS_FOLDER'], '.versions')

# Status codes
BAD_REQUEST = 400
NOT_FOUND = 404
OK = 200

INPUT_SAMPLE = '''{
            "id": "488ef383211f472f957337d38a909c43",
            "s1": 480.0,
            "s2": 105.2,
            "s3": 0.8514,
            "s4": 94.2
}'''


class CreditorTestCase(unittest.TestCase):

    """Independent tests for creditor module endpoints"""

    def setUp(self):
        app.testing = True
        self.app = app.test_client()
        test_model, auc = train(os.path.join(app.config['DATAFILES_FOLDER'], 'test_training_set.parquet'))
        test_model_filepath = os.path.join(app.config['MODELS_FOLDER'], app.config['MAIN_MODEL'])
        joblib.dump(test_model, test_model_filepath)

        model_creation_time = datetime.fromtimestamp(os.stat(test_model_filepath)[stat.ST_CTIME])
        current_version = {'id': app.config['MAIN_MODEL_ID'], 'created_at': str(model_creation_time), 'auc': auc}
        update_versions(current_version)

    def testHealthcheck(self):

        """Tests if healthcheck endpoint is working and returning OK"""

        r = self.app.get('/healthcheck')
        r_dict = json.loads(r.data)
        self.assertEqual(r_dict['status'], 'success', 'Unhealthy server')
        self.assertTrue(r_dict['results'][0]['passed'], 'Not passed')

    def test_slashed_endpoints(self):

        """Tests if URLs containing slashes returns error 404 NOT FOUND"""

        f = open(os.path.join(app.config['DATAFILES_FOLDER'], 'test_training_set.parquet'), 'rb')
        files = {'file': f}
        try:
            update_req = self.app.post('/update/../foo', data=files)
        finally:
            f.close()
        self.assertEqual(update_req.status_code, NOT_FOUND, 'Not 404 for a slashed model at /update')

        predict_req = self.app.post('/predict/../foo', data=INPUT_SAMPLE)
        self.assertEqual(predict_req.status_code, NOT_FOUND, 'Not 404 for a slashed model at /predict')

    def test_schema_validation(self):

        """Tests if schema validation errors are thrown as expected"""

        input_data_missing_fields = '{"s3": 0.8514, "s4": 94.2}'
        input_data_extra_fields = '''{
            "id": "488ef383211f472f957337d38a909c43", "s1": 480.0,
            "s2": 105.2, "s3": 0.8514,
            "s4": 94.2, "s5": 22.3
        }'''
        input_data_wrong_type = '''{
            "id": "488ef383211f472f957337d38a909c43", "s1": 480.0,
            "s2": "foo", "s3": "bar",
            "s4": 94.2
        }'''
        r_missing = self.app.post('/predict', data=input_data_missing_fields)
        self.assertEqual(r_missing.status_code, BAD_REQUEST, 'Not 400 for missing fields in JSON')
        r_extra = self.app.post('/predict', data=input_data_extra_fields)
        self.assertEqual(r_extra.status_code, BAD_REQUEST, 'Not 400 for extra fields in JSON')
        r_wrong = self.app.post('/predict', data=input_data_wrong_type)
        self.assertEqual(r_wrong.status_code, BAD_REQUEST, 'Not 400 for type error in JSON')
        r = self.app.post('/predict', data=INPUT_SAMPLE)
        self.assertEqual(r.status_code, OK, 'Not 200 for a valid JSON')

    def test_predict_nonexistent_model(self):

        """Tests if endpoint /predict returns 404 NOT FOUND for a nonexistent model"""

        r = self.app.post('/predict/foo', data=INPUT_SAMPLE)
        self.assertEqual(r.status_code, NOT_FOUND, 'Not 404 for nonexistent model')

    def test_update_with_not_allowed_extension(self):

        """Tests if a POST request to /update with a file containing a not allowed extension
        returns 400 BAD REQUEST
        """

        f = open(os.path.join(app.config['DATAFILES_FOLDER'], 'not_allowed.extension'), 'rb')
        files = {'file': f}
        try:
            r = self.app.post('/update/notallowed', data=files)
        finally:
            f.close()
        self.assertEqual(r.status_code, BAD_REQUEST, 'Not 400 for file missing or not allowed extension')

    def test_update_existent_model(self):

        """Tests if a POST request to /update with an existent model_id returns 400 BAD REQUEST"""

        model_id = app.config['MAIN_MODEL'][:-4]
        f = open(os.path.join(app.config['DATAFILES_FOLDER'], 'test_training_set.parquet'), 'rb')
        files = {'file': f}
        try:
            r = self.app.post('/update/{}'.format(model_id), data=files)
        finally:
            f.close()
        self.assertEqual(r.status_code, BAD_REQUEST, 'Not 400 for using an existent model_id')

    def tearDown(self):
        os.remove(os.path.join(app.config['MODELS_FOLDER'], app.config['MAIN_MODEL']))
        with open(app.config['VERSION_CONTROL'], 'w') as f:
            f.truncate()


class MonolithicCreditorTestCase(unittest.TestCase):

    """Monolithic test for creditor module endpoints. The output of one step is considered for the
    execution of the next step. Its purpose is to test the entire pipeline at once
    """

    def setUp(self):
        self.app = app.test_client()
        test_model, auc = train(os.path.join(app.config['DATAFILES_FOLDER'], 'test_training_set.parquet'))
        test_model_filepath = os.path.join(app.config['MODELS_FOLDER'], app.config['MAIN_MODEL'])
        joblib.dump(test_model, test_model_filepath)

        model_creation_time = datetime.fromtimestamp(os.stat(test_model_filepath)[stat.ST_CTIME])
        model_id = app.config['MAIN_MODEL'][:-4]
        current_version = {'id': model_id, 'created_at': str(model_creation_time), 'auc': auc}
        update_versions(current_version)

    # Predicts using the main model
    def step1(self):
        r = self.app.post('/predict', data=INPUT_SAMPLE)
        r_dict = json.loads(r.data)
        self.assertEqual(len(r_dict), 2)
        self.assertEqual(r_dict['id'], '488ef383211f472f957337d38a909c43')
        self.assertAlmostEqual(r_dict['probability'], 0.15156206771062417)

    # Creates a new model called embla
    def step2(self):
        f = open(os.path.join(app.config['DATAFILES_FOLDER'], 'test_random_training_set.parquet'), 'rb')
        files = {'file': f}
        try:
            r = self.app.post('/update/embla', data=files)
        finally:
            f.close()
        r_dict = json.loads(r.data)
        self.assertEqual(r.status_code, OK)
        self.assertIn('up and running', r_dict['message'])
        self.assertIn('embla.pkl', os.listdir(app.config['MODELS_FOLDER']))
        self.assertIn('embla.parquet', os.listdir(app.config['DATAFILES_FOLDER']))

    # Checks if embla is added to the list of model versions
    def step3(self):
        r = self.app.get('/versions')
        r_dict = json.loads(r.data)
        model_ids = [model['id'] for model in r_dict]
        self.assertEqual(len(model_ids), 2)
        self.assertIn(u'askr', model_ids)
        self.assertIn(u'embla', model_ids)

    # Predicts using embla
    def step4(self):
        r = self.app.post('/predict/embla', data=INPUT_SAMPLE)
        r_dict = json.loads(r.data)
        self.assertEqual(len(r_dict), 2)
        self.assertEqual(r_dict['id'], '488ef383211f472f957337d38a909c43')
        self.assertAlmostEqual(r_dict['probability'], 0.47938034279039)

    def _steps(self):
        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))

    def tearDown(self):
        os.remove(os.path.join(app.config['DATAFILES_FOLDER'], 'embla.parquet'))
        os.remove(os.path.join(app.config['MODELS_FOLDER'], 'askr.pkl'))
        os.remove(os.path.join(app.config['MODELS_FOLDER'], 'embla.pkl'))
        with open(app.config['VERSION_CONTROL'], 'w') as f:
            f.truncate()


if __name__ == '__main__':
    unittest.main()
