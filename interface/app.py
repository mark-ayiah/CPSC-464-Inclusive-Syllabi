from flask import Flask, render_template, request, redirect, url_for, session
import requests
import os
from docx import Document

app = Flask(__name__)
app.secret_key = 'supersecretkey'


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for the main page (upload page)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload and parsing
@app.route('/validate', methods=['POST', 'GET'])
def validate():       
        
    session['books'] = [{'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald', 'isbn': '9780743273565'},
                   {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee', 'isbn': '9780061120084'},
                   {'title': '1984', 'author': 'George Orwell', 'isbn': '9780451524935'}]
    
    return render_template('validate.html', books=session['books'])
    # file = request.files['file']
    # category = request.form.get('categories')

    # if file and category:

    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(filepath)

    #     # Extract text from the uploaded Word document
    #     text = extract_text_from_docx(filepath)

        
    #     # Find books using the OpenLibrary API
    #     books = find_books_in_pdf(text)

        
    #     # Render the template with the list of books
    #     return render_template('validate.html', books=books)

@app.route('/edit-book', methods=['POST'])
def edit_book():
    updated_isbns = request.form.getlist('isbns')
    
    session['books'] = []
    
    for isbn in updated_isbns:
        title, author, isbn = search_book_by_isbn(isbn)
        if title:
            session['books'].append({'title': title, 'author': author, 'isbn': isbn})

    return render_template('validate.html', books=session['books'])

@app.route('/add-book', methods=['POST'])
def add_book():
    new_title = request.form.get('new-title')
    if new_title:
        title, author, isbn = search_book(new_title)
        if title:
            session['books'].append({'title': title, 'author': author, 'isbn': isbn})
    return render_template('validate.html', books=session['books'])
        

@app.route('/results', methods=['POST'])
def results(isbns):
    diversity = calc_diversity(isbns)
    suggestions = get_suggestions(isbns)
    return render_template('results.html', diversity=diversity, suggestions=suggestions)


# Function to extract text from a PDF file
def extract_text_from_docx(filepath):
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
            author = book['author_name'][0]
            isbn = book.get('isbn', ['N/A'])[0]
            return title, author, isbn
    return None, None, None

def search_book_by_isbn(isbn):
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        book = data.get(f"ISBN:{isbn}")
        if book:
            title = book['title']
            author = book['authors'][0]['name']
            
        # title = book['title']
        # author = book['author_name'][0]
            return title, author, isbn
    return None, None, None

# Function to find books in the parsed Word Doc 
def find_books_in_pdf(text):
    book_list = []
    lines = text.split('\n')
    for line in lines:
        title, author, isbn = search_book(line)
        if title:
            book_list.append({'title': title, 'author': author, 'isbn': isbn})
    return book_list

def calc_diversity(isbns):
    return 0.349

def get_suggestions(isbns):
    return ['Their Eyes Were Watching By God',  'Invisible Man' , 'Native Son']

if __name__ == '__main__':
    app.run(debug=True)
