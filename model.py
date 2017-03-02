import csv
import cv2

lines = []
with open("./data/driving_log.csv") as csvfile :
    reader = csv.reader(csvfile)
    for line in reader :
        lines.append(line)

images = []
measurements = []

for line in lines :
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    
