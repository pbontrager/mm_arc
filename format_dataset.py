# Data formatting code generated by Llama 405B

import json
import os
import copy

# Set the directory path and the key to remove
directory_path = 'ARC-AGI/data/training/'
key_to_remove = 'name'

# Iterate through all JSON files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        filepath = os.path.join(directory_path, filename)

        # Load the JSON data from the file
        with open(filepath, 'r') as file:
            data = json.load(file)

        # Remove the optional key if it exists
        if key_to_remove in data:
            del data[key_to_remove]

        # Check if the "test" value is a list with more than one element
        if 'test' in data and isinstance(data['test'], list) and len(data['test']) > 1:
            # Duplicate the file for each "test" value
            for i, test_value in enumerate(data['test']):
                new_data = copy.deepcopy(data)
                new_data['test'] = [test_value]
                new_filename = f"{filename[:-5]}_{i+1}.json"
                new_filepath = os.path.join(directory_path, new_filename)

                # Write the new JSON data to the duplicated file
                with open(new_filepath, 'w') as new_file:
                    json.dump(new_data, new_file)

                print(f"Duplicated file: {new_filename}")
            os.remove(filepath)
        else:
            with open(filepath, 'w') as file:
                json.dump(data, file)
