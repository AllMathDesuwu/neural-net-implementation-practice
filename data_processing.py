from my_exceptions import *

class Converter(object):
    def __init__(self):
        self.converters = list()
        self.maps = dict()

        def convert(self, input):
            data_list_raw = input.split(",")
            if len(data_list_raw) != len(self.converters):
                raise ArgLenError(data_list_raw, self.converters)
            example = list()
            for i in range(len(data_list_raw)):
                entry = data_list_raw[i]
                conv = self.converters[i]
                wrapped_entry = conv(entry)
                example = example + wrapped_entry #effectively append
            return example

#makes a one-hot encoded list-- pos indicated the index of the one in the list, len indicates the length of the list-- all other positions are zero
def make_onehot_list(pos, len):
    oh_list = [0] * len
    oh_list[pos] = 1
    return oh_list

#wraps data in a list
def wrap(data):
    return [data]


#single-pass reader for header file-- considerably faster but requires information about unordered string input
#returns a Converter object-- this object will be used to convert data read from data file into usable objects
#IMPORTANT NOTE: the converters will wrap the data entry in a LIST-- this is intentional
#this is because when the file is being read line by line, each column will be added to the end of a list in the following fashion:
#[a, b, c] + [d] = [a, b, c, d]
def read_header_single_pass(file):
    with file:
        header = file.readline()
        header = header.replace(" ", "") #remove all whitespace
        header = header.split("}")
        for head_entry in header:
            if head_entry[0] != "{" and len(head_entry) != 0:
                raise NoHeaderError(file)
            head_entry = head_entry[1:] #this will delete the first char of the sequence-- which SHOULD be an opening bracket
        header.pop() #a properly formatted header will have one empty string at the very end after splitting
        if len(header) == 1: #your header definitely isn't formatted properly if there's only one column
            raise NoHeaderError(file)
        
        converter = Converter() #this will provide a list of functions for converting each entry of the header to its appropriate data type
        for i in range(len(header)):
            info = header[i].split(";")[-1] #extract the information portion-- the column name exists solely for the sake of human readability
            info = info.split(":")
            #now we'll determine the data type
            if info[0] == "num" or info[0] == "double" or info[0] == "float" or info[0] == "d" or info[0] == "f" or info[0] == "n": #num will just broadly be any number-- will be converted to double
                converter.converters[i] = lambda a: wrap(float(a))
            elif info[0] == "int":
                converter.converters[i] = lambda a: wrap(int(a))
            elif info[0] == "string_ordered" or info[0] == "str_ord" or info[0] == "so": #ordered string-- basically an enum or dict
                keys = info[1].split(",")
                converter.maps[i] = dict()
                the_map = converter.maps[i]
                for j in range(len(keys)):
                    the_map[keys[j]] = j
                converter.converters[i] = lambda a: wrap(converter.maps[i].get(a))
            elif info[0] == "string_unordered" or info[0] == "str_unord" or info[0] == "su": #unordered string-- this'll be handled using one-hot encoding
                keys = info[1].split(",")
                converter.maps[i] = dict()
                the_map = converter.maps[i]
                for j in range(len(keys)):
                    the_map[keys[j]] = j
                converter.converters[i] = lambda a: make_onehot_list(converter.maps[i].get(a), len(keys))

        return converter            