from student import Student

dspFmtWithNum = '| {0:^4} | {1:^6} | {2:16} | {3:^4} |'

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
    return Student(ID, name, age)

def listStudents(students):
    for i, student in enumerate(students):
        print(dspFmtWithNum.format(i+1, student.ID, student.name, student.age))

def deleteStudent(id, students):
    for i, student in enumerate(students):
        if student.ID == id:
            del students[i]
            return True
    return False


def main(students=[]):
    while True:
        choice = menu()
        if choice == 1: # View all students
            print("====== View all students ======")
            # print header
            print('=============================================')
            print(dspFmtWithNum.format("No.", "ID", "Name", "Age"))
            print('=============================================')
            # print data
            listStudents(students)
            print('=============================================')
            print()
        elif choice == 2: # Add a new student
            print("====== Add a new student ======")
            students.append(createStudent())
            print("Student added!")
            print()
        elif choice == 3: # Delete a student
            print("====== Delete a student ======")
            input_id = int(input("Enter ID to delete: "))
            if deleteStudent(input_id, students):
                print(f"Student {input_id} is deleted!")
            else:
                print("Student not found!")
            print()
        elif choice == 4: # Quit
            print("====== Quit ======")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    students = [
        Student(111, "Alice", 20),
        Student(112, "Bob", 21),
        Student(113, "Charlie", 22)
    ]
    main(students=students)