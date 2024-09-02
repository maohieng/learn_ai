from abc import ABC, abstractmethod
import json

# Defining the interface using an Abstract Base Class
class Dao(ABC):

    @abstractmethod
    def store_one(self, data):
        '''Store data into storage (file or database)'''
        pass

    @abstractmethod
    def store_all(self, datas):
        '''Store all data into storage (file or database)'''

    @abstractmethod
    def fetch(self):
        '''Fetch data from storage (file or database)'''
        pass

    @abstractmethod
    def delete(self, id):
        pass

class Filer(ABC):
    filename: str
    
    def __init__(self, filename: str):
        self.filename = filename

    @abstractmethod
    def data_to_text(self, data):
        '''Convert data obj to text'''
        pass

    @abstractmethod
    def text_to_data(self, text):
        '''Convert text to List of data obj'''
        pass

    @abstractmethod
    def write_text(self, text):
        '''Write text to file'''
        pass

    @abstractmethod
    def append_text(self, text):
        '''Append text to file'''
        pass

    @abstractmethod
    def read_text(self):
        '''Read text from file'''
        pass

    @abstractmethod
    def read_lines(self):
        '''Read all lines from text file'''
        pass

class TextFiler(Filer):
    '''Plain text file operations'''

    def __init__(self, filename: str):
        super().__init__(filename)

    def write_text(self, text):
        with open(self.filename, 'w') as f:
            f.write(text)
            f.write('\n')

    def append_text(self, text):
        with open(self.filename, 'a') as f:
            f.write(text)
            f.write('\n')

    def read_text(self):
        with open(self.filename, 'r') as f:
            return f.read()
    
    def read_lines(self):
        with open(self.filename, 'r') as f:
            return f.readlines()

class JsonFiler(Filer):
    '''Json file operations'''

    def __init__(self, filename: str):
        super().__init__(filename)
        
    def data_to_text(self, data):
        '''Converts obj data to json text (dict) by calling to_dict() method.
        Make sure your class implements to_dict() method.'''
        if isinstance(data, list):
            return json.dumps([d.to_dict() for d in data])
        
        return json.dumps(data)
    
    def text_to_data(self, text):
        '''Returns a list of dict data loaded from json text. 
        Make sure to convert it to your class object.'''
        return json.loads(text)
    
    def write_text(self, text):
        '''Write json text to file. Text must be a list of dict.'''
        with open(self.filename, 'w') as f:
            json.dump(text, f, indent=4, ensure_ascii=False)

    def read_text(self):
        '''Read json text from file and return as list of dict.'''
        with open(self.filename, 'r') as f:
            return json.load(f)
    
    def append_text(self, text):
        pass

    def read_lines(self):
        pass

class TextFileDao(Dao, TextFiler):
    
    def __init__(self, filename):
        super().__init__(filename)

    def store_one(self, data):
        text = self.data_to_text(data)
        try:
            self.append_text(text)
        except FileNotFoundError:
            self.write_text(text)
            
    def store_all(self, datas):
        # store json file
        for data in datas:
            self.store_one(data)

    def fetch(self):
        # Only fetch from text file
        text = self.read_text()
        return self.text_to_data(text)

class JsonFileDao(Dao, JsonFiler):

    def __init__(self, filename, id_fieldname=None):
        super().__init__(filename)
        self.id_fieldname = id_fieldname
    
    def store_one(self, data):
        # Store json file
        try:
            # read
            dicts = self.read_text()

            dataDict = data.to_dict()
            
            # Check if fieldname is not None
            if self.id_fieldname:
                # Check if data already exists
                for d in dicts:
                    if d[self.id_fieldname] == dataDict[self.id_fieldname]:
                        raise ValueError(f"Data with {self.id_fieldname} {dataDict[self.id_fieldname]} already exists")

            # append
            dicts.append(dataDict)
            # write
            self.write_text(dicts)
        except FileNotFoundError:
            self.write_text([data.to_dict()])
    
    def store_all(self, datas):
        # store json file
        for data in datas:
            self.store_one(data)
    
    def fetch(self):
        '''Fetch data from json file and return as list of dict.'''
        # Only fetch from json file
        return self.read_text()

    def delete(self, id):
        '''Delete data from json file'''
        if not self.id_fieldname:
            raise ValueError("ID fieldname is not defined")
        
        # read
        dicts = self.read_text()
        # filter
        dicts = [d for d in dicts if d[self.id_fieldname] != id]
        # write
        self.write_text(dicts)

class TextAndJsonFileDao(TextFileDao):
    
    def __init__(self, filename, jsonfile, id_fieldname='ID'):
        super().__init__(filename)
        # Json file operations becomes composition of this class
        self.jsonDao = JsonFileDao(jsonfile, id_fieldname)

    def store_one(self, data):
        # Store text file
        super().store_one(data)

        # Store json file
        self.jsonDao.store_one(data)