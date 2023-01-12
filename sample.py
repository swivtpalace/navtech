from flask import Blueprint, Flask, render_template,redirect,request, jsonify, make_response

flsk = Blueprint('flsk', __name__, template_folder='template')

@flsk.get('/')
def index():
    return '500'