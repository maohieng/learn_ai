from student import Student

dspFmtWithNum = '| {0:^4} | {1:^6} | {2:16} | {3:^4} |'

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
    return Student(ID, name, age)

def listStudents(students):
    for i, student in enumerate(students):
        print(dspFmtWithNum.format(i+1, student.ID, student.name, student.age))

def main(students=[]):
    while True:
        choice = menu()
        if choice == 1: # Create a student
            print("====== Create a student ======")
            students.append(createStudent())
            print("Student created!")
            print()
        elif choice == 2: # List students
            # print header
            print('================== List students ==================')
            print(dspFmtWithNum.format("No.", "ID", "Name", "Age"))
            print('===================================================')
            # print data
            listStudents(students)
            print('===================================================')
            print()
        elif choice == 3: # Quit
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