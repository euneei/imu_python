import os

class FileSplitter:
    chunk_size = 301
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def split_file(self, input_file, chunk_size=chunk_size):
        base_name, _ = os.path.splitext(os.path.basename(input_file))
        with open(input_file, 'r') as infile:
            file_content = infile.readlines()

        num_chunks = (len(file_content) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(file_content))

            output_file = os.path.join(self.base_dir, f"{base_name}_{str(i+1).zfill(2)}.txt")
            
            with open(output_file, 'w') as outfile:
                outfile.writelines(file_content[start:end])
            print(f"Chunk {i+1} has been written to {output_file}")


