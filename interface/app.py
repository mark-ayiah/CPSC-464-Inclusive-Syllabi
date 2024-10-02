from flask import Flask, render_template, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    print("HELLO")
    return render_template('index.html', show_results=False)

@app.route('/submit', methods=['POST'])
def submit():
    isbns = request.form.get('isbns').split(',')
    print(f"ISBN: {isbns}")
    return render_template('validate.html', isbns=isbns)



@app.route('/validate', methods=['POST'])
def validate():
    
    
    return render_template('results.html')

@app.route('/results', methods=['POST'])
def results(isbns):
    diversity = calc_diversity(isbns)
    suggestions = get_suggestions(isbns)
    return render_template('results.html', diversity=diversity, suggestions=suggestions)

def calc_diversity(isbns):
    return 0.349

def get_suggestions(isbns):
    return ['Their Eyes Were Watching By God',  'Invisible Man' , 'Native Son']

if __name__ == '__main__':
    app.run(debug=True)
