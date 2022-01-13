import chardet

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        text = f.read()
        predict =  chardet.detect(text)
    return predict['encoding']


def convert_file_encoding(file_path):
    with open('output.csv', 'wb') as output:
        with open(file_path, 'rb') as file:
            for line in file.readlines():
                line = line.decode('big5').encode('utf8')

                output.write(line)


if __name__ == '__main__':
    file_path = 'hw3/train.csv'
    print(detect_file_encoding(file_path))
    #convert_file_encoding(file_path)