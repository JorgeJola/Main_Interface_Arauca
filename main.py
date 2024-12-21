from flask import Blueprint, Flask,render_template

main=Blueprint('main',__name__)

@main.route('/')
def main_interface():
    return render_template('main_interface.html')

