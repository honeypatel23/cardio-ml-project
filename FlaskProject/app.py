from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    age = request.form['age']
    return f"Hello {name}, you are {age} years old!"

if __name__ == '__main__':
    app.run(debug=True)
