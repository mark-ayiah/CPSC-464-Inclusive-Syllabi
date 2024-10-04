from flask import Flask, render_template, request, redirect, url_for
import requests
import os
from docx import Document

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for the main page (upload page)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload and parsing
@app.route('/validate', methods=['POST'])
def submit():
    print("gang")

    file = request.files['file']
    category = request.form.get('categories')

    print("working")

    if file and category:

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text from the uploaded Word document
        if file.filename.endswith('.docx'):
            text = extract_text_from_docx(filepath)
        else:
            return "Unsupported file type. Please upload a .docx file."
        
        # Find books using the OpenLibrary API
        books = find_books_in_pdf(text)

        print(books)
        
        # Render the template with the list of books
        return render_template('validate.html', books=books)

# Function to extract text from a PDF file
def extract_text_from_docx(filepath):
    print("test")
    doc = Document(filepath)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to search for books using OpenLibrary API
def search_book(line):
    url = "http://openlibrary.org/search.json"
    params = {'q': line}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['numFound'] > 0:
            book = data['docs'][0]  # Assuming the first result is the most relevant
            title = book['title']
            isbn = book.get('isbn', ['N/A'])[0]
            print(f"Title: {title}\n")
            print(f"ISBN: {isbn}\n")
            return title, isbn
    return None, None

# Function to find books in the parsed Word Doc 
def find_books_in_pdf(text):
    book_list = []
    lines = text.split('\n')
    for line in lines:
        print(f"Line:{line}")
        title, isbn = search_book(line)
        if title:
            book_list.append({'title': title, 'isbn': isbn})
    return book_list

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
