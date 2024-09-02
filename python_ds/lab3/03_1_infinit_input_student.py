from student_dao import StudentFileDao
from student import Student

dspFmtWithNum = '| {0:^4} | {1:^6} | {2:16} | {3:^4} |'

studentDao = StudentFileDao('students.txt', 'students.json')

def menu():
    print("====== Menu ======")
    print("1. Create a student")
    print("2. List students")
    print("3. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createStudent():
    ID = int(input("Enter ID: "))
    name = input("Enter name: ")
    age = int(input("Enter age: "))
    s = Student(ID, name, age)
    # Store the student
    studentDao.store_one(s)
    return s

def listStudents():
    # Fetch students
    students = studentDao.fetch()
    # print students
    for i, student in enumerate(students):
        print(dspFmtWithNum.format(i+1, student.ID, student.name, student.age))

def main():
    while True:
        choice = menu()
        if choice == 1: # Create a student
            print("====== Create a student ======")
            try:
                createStudent()
            except ValueError as e:
                print(e)
                print()
                continue
            
            print("Student created!")
            print()
        elif choice == 2: # List students
            # print header
            print('================== List students ==========')
            print(dspFmtWithNum.format("No.", "ID", "Name", "Age"))
            print('===========================================')
            # print data
            listStudents()
            print('===========================================')
            print()
        elif choice == 3: # Quit
            print("====== Quit ======")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    # Check if data doesn't exist, create some
    try:
        studentDao.fetch()
    except FileNotFoundError:
        students = [
            Student(111, "Alice", 20),
            Student(112, "Bob", 21),
            Student(113, "Charlie", 22)
        ]
        studentDao.store_all(students)
    
    main()