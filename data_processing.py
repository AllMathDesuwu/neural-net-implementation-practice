from my_exceptions import *

def read_header(file):
    with file:
        header = file.readline()
        header = header.replace(" ", "") #remove all whitespace
        header = header.split("}")
        for head_entry in header:
            if head_entry[0] != "{" and len(head_entry) != 0:
                raise(NoHeaderError(file))
            head_entry = head_entry[1:] #this will delete the first char of the sequence-- which SHOULD be an opening bracket
        header.pop() #a properly formatted header will have one empty string at the very end after splitting
        if len(header) == 1: #your header definitely isn't formatted properly if there's only one column
            raise(NoHeaderError(file))
        
        converters = list() #this will be a list of functions for converting each entry of the header to its appropriate data type
        for entry in header:
            info = entry.split(";")[-1] #get rid of the column name-- this exists solely for the sake of human readability






