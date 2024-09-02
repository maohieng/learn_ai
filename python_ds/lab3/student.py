from typing import Any

from department import Department

displayFmt = '| {0:^6} | {1:^20} | {2:^4} |'

class Student:
    ID: int
    name: str
    age: int
    department: Department

    def __init__(self, ID: int, name: str, age: int, department: Department = None):
        self.ID = ID
        self.name = name
        self.age = age
        self.department = department
    
    def __str__(self):
        return f'{self.ID} {self.name}'
    
    def display(self):
        print(displayFmt.format(self.ID, self.name, self.age))
    
    def __eq__(self, other: Any) -> bool:
        return self.ID == other.ID
    
    def setDepartment(self, department: Department):
        self.department = department

    def to_dict(self):
        return {
            'ID': self.ID,
            'name': self.name,
            'age': self.age,
            'department': self.department.name if self.department else ''
        }
    
    @staticmethod
    def from_dict(data):
        s = Student(ID=data['ID'], name=data['name'], age=data['age'])
        if data['department']:
            s.department = Department(data['department'])
        return s