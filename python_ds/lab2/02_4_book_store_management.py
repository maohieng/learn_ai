from typing import List
from book import Book

dspFmtWithNum = '| {0:^4} | {1:^6} | {2:28} | {3:^4} | {4:^12} |'

def menu():
    print("====== Menu ======")
    print("1. View all books")
    print("2. Add a new book")
    print("3. Update a book")
    print("4. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createBook() -> Book:
    ID = int(input("Enter ISBN: "))
    name = input("Enter title: ")
    price = float(input("Enter price: "))
    author = input("Enter author: ")
    return Book(ID, name, price, author)

def listBooks(books: List[Book]) -> None:
    for i, b in enumerate(books):
        print(dspFmtWithNum.format(i+1, b.ISBN, b.title, b.price, 'unknown' if not b.author else b.author))

def findBook(id: int, books: List[Book]):
    for book in books:
        if book.ISBN == id:
            return book
    return None

def updateBook(id: int, books: List[Book]) -> bool:
    book = findBook(id, books)
    if book:
        print("You are about to update book with ISBN:", id)
        title = input(f"Enter new title ({book.title}): ")
        price = float(input(f"Enter new price ({book.price}): "))
        author = input(f"Enter new author ({book.author}): ")
        
        if title and title != book.title:
            book.title = title
        
        if price and price != book.price:
            book.price = price
        
        if author and author != book.author:
            book.author = author

        return True
    return False


def main(books=[]):
    while True:
        choice = menu()
        if choice == 1: # View all
            print("====== View all books ======")
            # print header
            print('=============================================')
            print(dspFmtWithNum.format("No.", "ISBN", "Title", "Price", "Author"))
            print('=============================================')
            # print data
            listBooks(books)
            print('=============================================')
            print()
        elif choice == 2: # Add new
            print("====== Add a new book ======")
            books.append(createBook())
            print("book added!")
            print()
        elif choice == 3: # Update book
            print("====== Update a book ======")
            input_id = int(input("Enter ISBN to update: "))
            if updateBook(input_id, books):
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
    books = [
        Book(ISBN=111, title="Python Programming Language", price=20, author="Alice"),
        Book(ISBN=112, title="Java Programming Language", price=25, author="Bob"),
        Book(ISBN=113, title="C Programming Language", price=15, author="Charlie"),
    ]
    main(books=books)