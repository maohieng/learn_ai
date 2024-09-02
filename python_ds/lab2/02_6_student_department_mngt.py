from functools import reduce
from typing import List
from department import Department
from student import Student

dspFmtWithNum = '| {0:^3} | {1:^6} | {2:16} | {3:^3} | {4:^10} |'

def menu():
    print("====== Menu ======")
    print("1. View all students")
    print("2. View all departments")
    print("3. Add a new student")
    print("4. Delete a student")
    print("5. Quit")
    print("==================")
    return int(input("Enter your choice: "))

def createStudent(deps: List[Department], students: List[Student] = []) -> Student:
    # print("Available departments:", list(map(lambda d: d.name, deps)))
    
    foundStudent = None
    while True:
        id = int(input("Enter ID: "))
        # TODO This existing check doesn't work
        for s in students:
            if s.ID == id:
                foundStudent = s
                break

        if foundStudent is not None:
            print(f"Student ID {id} already exists! ({foundStudent.name})")
            continue
        else:
            break

    name = input("Enter name: ")
    age = int(input("Enter age: "))

    foundDep = None
    while True:
        dep = input("Enter department: ")
        # Find department object
        for d in deps:
            if d.name.lower() == dep.lower():
                foundDep = d
                break

        if foundDep is None:
            print("Department not found! Available:", list(map(lambda d: d.name, deps)))
            continue
        else:
            break
    
    s = Student(id, name, age, foundDep)
    return s

def listStudents(students: List[Student]):
    for i, student in enumerate(students):
        print(dspFmtWithNum.format(i+1, student.ID, student.name, student.age, student.department.name if student.department is not None else "None"))

def deleteStudent(id, students):
    for i, student in enumerate(students):
        if student.ID == id:
            del students[i]
            return True
    return False


def main(deps=[], students=[]):
    while True:
        choice = menu()
        if choice == 1: # View all students
            # print header
            print('==================== View all students ====================')
            print(dspFmtWithNum.format("No.", "ID", "Name", "Age", "Department"))
            print('===========================================================')
            # print data
            listStudents(students)
            print('===========================================================')
            print()
        elif choice == 2: # View all departments
            print("========= View all departments =========")
            print("Departments:")
            for i, dep in enumerate(deps):
                print(f"{i+1}. {dep.name}")
            print()
        elif choice == 3: # Add a new student
            print("========= Add a new student =========")
            students.append(createStudent(deps=deps))
            print("Student added!")
            print()
        elif choice == 4: # Delete a student
            print("========= Delete a student =========")
            input_id = int(input("Enter ID to delete: "))
            if deleteStudent(input_id, students):
                print(f"Student {input_id} is deleted!")
            else:
                print("Student not found!")
            print()
        elif choice == 5: # Quit
            print("========= Quit =========")
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")
            continue    

if __name__ == '__main__':
    deps = [
        Department("GIC"),
        Department("GEC"),
        Department("GCC")
    ]
    
    students = [
        Student(111, "Alice", 20),
        Student(112, "Bob", 21),
        Student(113, "Charlie", 22)
    ]
    main(deps=deps, students=students)