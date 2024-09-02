class Book:
    ISBN: int
    title: str
    price: float
    author: str

    def __init__(self, ISBN: int, title: str, price: float, author: str):
        self.ISBN = ISBN
        self.title = title
        self.price = price
        self.author = author
    
    def __str__(self):
        return self.title
    
