import csv
import numpy as np

def read_csv_and_split(filename, row_splits):
    """
    Reads a CSV file and splits it into multiple numpy arrays based on specified row splits.

    Args:
        filename: The name of the CSV file.
        row_splits: A list of integers indicating the number of rows for each split.

    Returns:
        A list of numpy arrays, each representing a split portion of the CSV data.
    """

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        # Initialize empty lists for each split
        splits = [[] for _ in row_splits]

        # Loop through the CSV rows and distribute them to the appropriate splits
        for row in csv_reader:
            numeric_values = [float(value) for value in row[1:]]  # Convert the rest to float4
            n = len(numeric_values)
            if n == 100 :
                splits[0].append(numeric_values)
            elif n == 40:
                splits[1].append(numeric_values)
            elif n == 4:
                splits[2].append(numeric_values)

        # Convert each split to a numpy array
        return [np.array(split) for split in splits]


        for row in csv_reader:
            numeric_values = [float(value) for value in row[1:]]  # Convert the rest to float
            for i, split in enumerate(splits):
                if len(split) < row_splits[i]:
                    split.append(numeric_values)
                else:
                    break

        # Convert each split to a numpy array
        return [np.array(split) for split in splits]

# Example usage
filename = 'Task_1/a/w.csv'
row_splits = [14, 100, 40]

w1, w2, w3 = read_csv_and_split(filename, row_splits)

print("w1:", w1)
print("w1.shape:", w1.shape)
print("w2:", w2)
print("w2.shape:", w2.shape)
print("w3:", w3)
print("w3.shape:", w3.shape)