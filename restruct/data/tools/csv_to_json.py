import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Dictionary to hold the JSON structure
    json_data = {"_via_img_metadata": {}}

    # Read the CSV file
    with open(csv_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)

        # Iterate over each row in the CSV
        for row in csv_reader:
            filename = row['filename']
            file_size = int(row['file_size'])
            file_attributes = json.loads(row['file_attributes'])
            region_shape_attributes = json.loads(row['region_shape_attributes'].replace("''", "\""))
            region_attributes = json.loads(row['region_attributes'].replace("''", "\""))

            # If the filename is not already a key in the dictionary, add it
            if filename not in json_data["_via_img_metadata"]:
                json_data["_via_img_metadata"][filename] = {
                    "filename": filename,
                    "size": file_size,
                    "regions": [],
                    "file_attributes": file_attributes
                }

            # Append the region to the list of regions for this filename
            json_data["_via_img_metadata"][filename]["regions"].append({
                "shape_attributes": region_shape_attributes,
                "region_attributes": region_attributes
            })

    # Write the JSON data to a file
    with open(json_file_path, mode='w') as jsonfile:
        json.dump(json_data, jsonfile, indent=4)

# Example usage
csv_file_path = 'new_dataset/annotations.csv'
json_file_path = 'new_dataset/converted_annotations.json'
csv_to_json(csv_file_path, json_file_path)
