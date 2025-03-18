import glob
import os
import pandas as pd
import json
# Spécifiez le chemin du répertoire
directory = 'test/'
annotations = 'test/bh-annotation.csv'
output_directory = 'result/annotations.csv'
prefix = 'bh'

# Liste tous les fichiers dans le répertoire
files = glob.glob(os.path.join(directory, '*'))
files = [os.path.basename(file) for file in files]

annotations = pd.read_csv(annotations)

file = "0000.jpg"

file_annotations = annotations[annotations['filename'] == file]

for index, row in file_annotations.iterrows():

    image_original_size = (1024, 1024)
    image_new_size = (512, 683)

    print(row['filename'].split('.')[0] + '.png')

    region_shape_attributes = json.loads(row['region_shape_attributes'])
    print(region_shape_attributes['all_points_x'])