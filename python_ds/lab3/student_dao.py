import logging
from dao import TextAndJsonFileDao
from department import Department
from student import Student

class StudentFileDao(TextAndJsonFileDao):

    def __init__(self, textfile, jsonfile):
        super().__init__(filename=textfile, jsonfile=jsonfile, id_fieldname='ID')

    def data_to_text(self, data):
        return f"{data.ID},{data.name},{data.age},{data.department.name if data.department else ''}"

    def text_to_data(self, text):
        lines = text.split('\n')
        students = []
        for line in lines:
            if line:
                ID, name, age,dep = line.split(',')
                s = Student(int(ID), name, int(age))
                if dep:
                    s.setDepartment(Department(dep))
                students.append(s)
        return students

    def store_one(self, data):
        logging.info(f"StudentRepo: storing {data}")
        try:
            # Check existing data
            lines = self.read_lines()
            for line in lines:
                if int(line.split(',')[0]) == data.ID:
                    raise ValueError(f"Student with ID {data.ID} already exists")
        except FileNotFoundError:
            pass
        
        super().store_one(data)

    def store_all(self, datas):
        logging.info(f"StudentRepo: storing all")
        return super().store_all(datas)

    def fetch(self):
        logging.info(f"StudentRepo: fetching")
        return super().fetch()
        

    def delete(self, id):
        logging.info(f"StudentRepo: deleting {id}")
        # delete from text file
        lines = self.read_lines()
        with open(self.filename, 'w') as f:
            for line in lines:
                if int(line.split(',')[0]) != id:
                    f.write(line)
        
        # delete from json file
        self.jsonDao.delete(id)
