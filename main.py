from preprocessing import Preprocessor
from cut_file import FileSplitter
import glob 

data_directory = '/path/to/your/data'

splitter = FileSplitter(data_directory)

# cut one file
input_file_path = '/path/to/your/data_one_file.txt' 
splitter.split_file(input_file_path) 

# cut multiple file
input_file_path2 = glob.glob('/path/to/your/data_file/*.txt') # ex) data_directory/walk/*.txt
for file_path in input_file_path2:
    splitter.split_file(input_file_path2)


labels = {
    'walk/*.txt': 0, 
    'run/*.txt': 1,
    'dan/*.txt': 2,
    'nor/*.txt' : 3
}
concat_file = Preprocessor(data_directory)
concat_file.process_all_data(labels)
