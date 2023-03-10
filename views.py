# CONFIDENTIAL 
# This codes are is intended only for the use of the individual or entity to which it is addressed. It is a classified information that is privileged and confidential.
import time
from flask import Flask, render_template, redirect, request, jsonify, make_response, Response, url_for
import glob
import json
from functools import wraps
import sys
#sys.path.insert(1, '/root/jFiles')
import ver
import os
from flask_cors import CORS, cross_origin
import resource
from json import dumps
import pickle
from worker.worker import make_celery
import random
from celery.result import AsyncResult
from werkzeug.exceptions import HTTPException
import uuid
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1,'/root/jFiles')
#import ver

''' 
 Flask server and celery application main configuration 
'''
REDIS_BROKER_URL = 'redis://localhost:6379/0'
app = Flask(__name__)
app.config.update(CELERY_CONFIG={
    'broker_url': REDIS_BROKER_URL,
    'result_backend': REDIS_BROKER_URL,
})
celery_worker = make_celery(app)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/upload/*": {"origins":  "*"}})
cors = CORS(app, resources={r"/API/*": {"origins": "*"}})

SERVER_PASSWORD = 'ABC!@#123'

'''---------   Main Configuration end ----------------'''

''' Define celery task for heavy loads  '''


@celery_worker.task()
def process_image_processing(train_dir, is_not_test_mode=True):
    json_dump = ver.get_loc(train_dir=train_dir, is_not_test_mode=is_not_test_mode)
    return {"status": True, "result": json_dump}


''' End celery tasks '''


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', defaults={'task_id': ''})
@app.route("/upload/<task_id>")
def upload(task_id):
    return render_template('upload.html', task_id=task_id)


def _upload_task(req):
    password = req.form.get('password')
    if password != SERVER_PASSWORD:
        return "Unauthorised request"
    if 'files[]' not in req.files:
        return "Please upload images"
    files = req.files.getlist('files[]')
    photo_folder_name = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(10))
    train_path = f'photos/{photo_folder_name}'
    os.mkdir(train_path)
    for file in files:
        filename = file.filename
        file.save(os.path.join(train_path, filename))
    task = process_image_processing.delay(photo_folder_name)
    return task


@app.route('/uploads', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        task = _upload_task(request)
        return redirect(url_for(".upload", task_id=task.id))


@app.route('/file/upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        task = _upload_task(request)
        if hasattr(task, 'id'):
            task.wait()
            job_result = AsyncResult(task.id, app=celery_worker)
            return job_result.result
        else:
            return {'message': 'Task object not define'}
    return render_template('file-upload.html')


@app.route('/api/uploads', methods=['POST'])
def get_api_upload():
    if request.method == 'POST':
        task = _upload_task(request)
        if hasattr(task, 'id'):
            task.wait()
            job_result = AsyncResult(task.id, app=celery_worker)
            return job_result.result
        else:
            return {'message': 'Task object not define'}


@app.route('/test-workload')
def get_workload_test():
    train_paths = ['photo1', 'photo2', 'photo3', 'photo4', 'photo5']
    task = process_image_processing.delay(random.choice(train_paths), False)
    task.wait()
    job = AsyncResult(task.id, app=celery_worker)
    return {
        "state": job.state,
        "result": job.result
    }


@app.route('/progress/<task_id>', methods=['GET'])
def get_async_result(task_id):
    res = dict()
    if task_id:
        job = AsyncResult(task_id, app=celery_worker)
        try:
            if job.state == 'PENDING':
                res = {
                    "state": job.state,
                    "result": None
                }
            elif job.state == 'SUCCESS':
                res = {
                    "state": job.state,
                    "result": job.result
                }
        except HTTPException:
            res = {
                "state": 'exception',
                "result": None
            }
    else:
        res = {
                "state": 'no_task_id',
                "result": None
            }
    return jsonify(res), 200


@app.route("/clean_image")
def clean_image():
    # for removing all the files from the /root/jFiles/NT_JWL_Model/img_folder
    dir_path = 'photo'
    i = 0
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        i = i + 1
    return "Files in img_folder are deleted"


if __name__ == 'main':
    app.run(host="0.0.0.0", debug=True, port=5000)

