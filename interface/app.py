from flask import Flask, render_template, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', show_results=False)

@app.route('/submit', methods=['POST'])
def submit():
    isbns = request.form.get('isbns').split(',')
    print(f"ISBN: {isbns}")
    return render_template('index.html', diversity=calc_diversity(isbns), suggestions=get_suggestions(isbns), show_results=True, isbns=','.join(isbns))

def calc_diversity(isbns):
    return 0.349

def get_suggestions(isbns):
    return ['To Kill a Mocking Bird', 'Autobiography of Malcolm X', '1984' , "Harry Potter"]

if __name__ == '__main__':
    app.run(debug=True)
