import logging
from dao import TextAndJsonFileDao
from book import Book

class BookFileDao(TextAndJsonFileDao):

    def __init__(self, textfile, jsonfile):
        super().__init__(filename=textfile, jsonfile=jsonfile, id_fieldname='ISBN')

    def data_to_text(self, data):
        return f"{data.ISBN},{data.title},{data.price},{data.author}"

    def text_to_data(self, text):
        lines = text.split('\n')
        books = []
        for line in lines:
            if line:
                ISBN, title, price, author = line.split(',')
                s = Book(int(ISBN), title, float(price), author)
                books.append(s)
        return books

    def store_one(self, data):
        logging.info(f"BookRepo: storing {data}")
        try:
            # Check existing data
            lines = self.read_lines()
            for line in lines:
                if int(line.split(',')[0]) == data.ISBN:
                    raise ValueError(f"Book with ISBN {data.ISBN} already exists")
        except FileNotFoundError:
            pass
        
        super().store_one(data)

    def store_all(self, datas):
        logging.info(f"BookRepo: storing all")
        return super().store_all(datas)

    def fetch(self):
        logging.info(f"BookRepo: fetching")
        # return super().fetch()
        # fetch from json file
        data = self.jsonDao.fetch()
        return [Book.from_dict(d) for d in data]
        

    def delete(self, id):
        logging.info(f"BookRepo: deleting {id}")
        # delete from text file
        lines = self.read_lines()
        with open(self.filename, 'w') as f:
            for line in lines:
                if int(line.split(',')[0]) != id:
                    f.write(line)
        
        # delete from json file
        self.jsonDao.delete(id)

    def getBook(self, id: int):
        books = self.fetch()
        for book in books:
            if book.ISBN == id:
                return book
        return None

    def updateBook(self, oldBook: Book, newBook: Book):
        if newBook.title and newBook.title != oldBook.title:
            oldBook.title = newBook.title
        
        if newBook.price and newBook.price != oldBook.price:
            oldBook.price = newBook.price
        
        if newBook.author and newBook.author != oldBook.author:
            oldBook.author = newBook.author

        self.delete(oldBook.ISBN)
        self.store_one(oldBook)

        