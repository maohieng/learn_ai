def write_text_to_file(text, filename):
    with open(filename, 'w') as f:
        f.write(text)
        f.write('\n')

def append_text_to_file(text, filename):
    with open(filename, 'a') as f:
        f.write(text)
        f.write('\n')

def read_text_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()
    
def read_lines_from_file(filename):
    with open(filename, 'r') as f:
        return f.readlines()