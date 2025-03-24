import csv
import json
import argparse

def json_to_csv(json_file_path, csv_file_path):
    # Read the JSON file
    with open(json_file_path, mode='r') as jsonfile:
        json_data = json.load(jsonfile)

    # Prepare CSV headers
    headers = ['filename', 'file_size', 'file_attributes', 'region_shape_attributes', 'region_attributes']

    # Write to CSV
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
        csv_writer.writeheader()

        # Iterate over each image in the JSON data
        for filename, image_data in json_data["_via_img_metadata"].items():
            file_size = image_data["size"]
            file_attributes = json.dumps(image_data["file_attributes"])

            # Write a row for each region
            for region in image_data["regions"]:
                row = {
                    'filename': filename,
                    'file_size': file_size,
                    'file_attributes': file_attributes,
                    'region_shape_attributes': json.dumps(region["shape_attributes"]),
                    'region_attributes': json.dumps(region["region_attributes"])
                }
                csv_writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convertir un fichier JSON en CSV')
    parser.add_argument('json_file', help='Chemin vers le fichier JSON d\'entrée')
    parser.add_argument('csv_file', help='Chemin vers le fichier CSV de sortie')
    
    args = parser.parse_args()
    json_to_csv(args.json_file, args.csv_file) 