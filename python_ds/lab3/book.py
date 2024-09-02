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
    
    def to_dict(self):
        return {
            'ISBN': self.ISBN,
            'title': self.title,
            'price': self.price,
            'author': self.author
        }
    
    @staticmethod
    def from_dict(data: dict):
        return Book(ISBN=data['ISBN'], title=data['title'], price=data['price'], author=data['author'])
