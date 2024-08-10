import json
import os

def move_camera_positions_closer(file_path, factor=0.5):
    print("Adjusting camera positions in", file_path)
    file_path = os.path.expanduser(file_path)

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for frame in data['frames']:
        transform_matrix = frame['transform_matrix']
        # Convert the string values to float
        transform_matrix = [[float(value) for value in row] for row in transform_matrix]
        
        print("Old transform matrix:", transform_matrix[0][3])
        
        # Adjust the translation components (last column of the matrix)
        transform_matrix[0][3] *= factor
        transform_matrix[1][3] *= factor
        transform_matrix[2][3] *= factor
        
        
        # Convert the float values back to string
        transform_matrix = [[str(value) for value in row] for row in transform_matrix]
        frame['transform_matrix'] = transform_matrix
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Usage
# move_camera_positions_closer('transforms_test.json')