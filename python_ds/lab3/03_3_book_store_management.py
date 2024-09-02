from typing import List
from book import Book
from book_dao import BookFileDao

dspFmtWithNum = '| {0:^3} | {1:^6} | {2:28} | {3:^5} | {4:12} |'

bookDao = BookFileDao('books.txt', 'books.json')

def menu():
    print("====== Menu ======")
    print("1. View all books")
    print("2. Add a new book")
    print("3. Update a book")
    print("4. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createBook():
    ID = int(input("Enter ISBN: "))
    name = input("Enter title: ")
    price = float(input("Enter price: "))
    author = input("Enter author: ")
    b = Book(ID, name, price, author)
    # Store the book
    bookDao.store_one(b)


def listBooks():
    # Fetch books
    books = bookDao.fetch()
    for i, b in enumerate(books):
        print(dspFmtWithNum.format(i+1, b.ISBN, b.title, b.price, 'unknown' if not b.author else b.author))

def updateBook(id: int) -> bool:
    book = bookDao.getBook(id)
    if book:
        print("You are about to update book with ISBN:", id)
        title = input(f"Enter new title ({book.title}): ")
        p = input(f"Enter new price ({book.price}): ")
        price = float(p if p else book.price)
        author = input(f"Enter new author ({book.author}): ")
        
        bookDao.updateBook(book, Book(ISBN=id, title=title, price=price, author=author))

        return True
    return False


def main():
    while True:
        choice = menu()
        if choice == 1: # View all
            print("====== View all books ======")
            # print header
            print('======================================================================')
            print(dspFmtWithNum.format("No.", "ISBN", "Title", "Price", "Author"))
            print('======================================================================')
            # print data
            listBooks()
            print('======================================================================')
            print()
        elif choice == 2: # Add new
            print("====== Add a new book ======")
            try:
                createBook()
            except ValueError as e:
                print(e)
                print()
                continue
            print("book added!")
            print()
        elif choice == 3: # Update book
            print("====== Update a book ======")
            input_id = int(input("Enter ISBN to update: "))
            if updateBook(input_id):
                print(f"book {input_id} is deleted!")
            else:
                print("book not found!")
            print()
        elif choice == 4: # Quit
            print("====== Quit ======")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    try:
        books = bookDao.fetch()
    except FileNotFoundError:
        books = [
            Book(ISBN=111, title="Python Programming Language", price=20, author="Alice"),
            Book(ISBN=112, title="Java Programming Language", price=25, author="Bob"),
            Book(ISBN=113, title="C Programming Language", price=15, author="Charlie"),
        ]
        bookDao.store_all(books)
    
    main()