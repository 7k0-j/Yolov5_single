
import time
def save_txt(file_name,content,index):
    date = time.strftime('%Y-%m-%d %H:%M:%S').split()
    with open(file_name,'w') as f:
        f.write(str(date))
        f.write('\n')
        f.write(index)
        f.write(':')
        f.write('\n')
        f.write(str(content))





