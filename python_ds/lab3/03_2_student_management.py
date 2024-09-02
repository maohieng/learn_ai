from student import Student
from student_dao import StudentFileDao

dspFmtWithNum = '| {0:^4} | {1:^6} | {2:16} | {3:^4} |'

studentDao = StudentFileDao('students.txt', 'students.json')

def menu():
    print("====== Menu ======")
    print("1. View all students")
    print("2. Add a new student")
    print("3. Delete a student")
    print("4. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createStudent():
    ID = int(input("Enter ID: "))
    name = input("Enter name: ")
    age = int(input("Enter age: "))
    studentDao.store_one(Student(ID, name, age))

def listStudents():
    # Fetch students
    students = studentDao.fetch()

    for i, student in enumerate(students):
        print(dspFmtWithNum.format(i+1, student.ID, student.name, student.age))

def deleteStudent(id):
    studentDao.delete(id)


def main():
    while True:
        choice = menu()
        if choice == 1: # View all students
            print("====== View all students ======")
            # print header
            print('=============================================')
            print(dspFmtWithNum.format("No.", "ID", "Name", "Age"))
            print('=============================================')
            # print data
            listStudents()
            print('=============================================')
            print()
        elif choice == 2: # Add a new student
            print("====== Add a new student ======")
            try:
                createStudent()
            except ValueError as e:
                print(e)
                print()
                continue
            
            print("Student added!")
            print()
        elif choice == 3: # Delete a student
            print("====== Delete a student ======")
            input_id = int(input("Enter ID to delete: "))
            deleteStudent(input_id)
            print(f"Student {input_id} is deleted!")
            
            print()
        elif choice == 4: # Quit
            print("====== Quit ======")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    main()