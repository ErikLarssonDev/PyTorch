import csv
import random

# Define the file names
input_csv_file = '../diabetic-retinopathy-detection/trainLabels.csv'
val_csv_file = '../diabetic-retinopathy-detection/val_labels.csv'
train_csv_file = '../diabetic-retinopathy-detection/train_labels.csv'

# Define the number of rows for the validation set
val_set_size = 4000

# Open the input CSV file and create output CSV files for validation and training sets
with open(input_csv_file, 'r') as input_file, \
        open(val_csv_file, 'w', newline='') as val_output_file, \
        open(train_csv_file, 'w', newline='') as train_output_file:

    # Initialize CSV writers for validation and training sets
    val_writer = csv.writer(val_output_file)
    train_writer = csv.writer(train_output_file)

    # Read the CSV header
    header = next(input_file)
    val_writer.writerow(header.split(','))
    train_writer.writerow(header.split(','))

    # Initialize counters for the validation set
    val_count = 0
    total_count = 0

    # Create a dictionary to hold image rows with the same XX
    image_dict = {}

    # Read the CSV rows and split them into validation and training sets
    for row in csv.reader(input_file):
        total_count += 1

        # Extract XX from the image name
        xx = row[0].split('_')[0]

        # Check if we have already encountered this XX, and if not, create a list for it
        if xx not in image_dict:
            image_dict[xx] = []

        # Add the row to the corresponding XX list
        image_dict[xx].append(row)

    # Shuffle the XX lists to randomize the order
    for xx in image_dict:
        random.shuffle(image_dict[xx])

    # Write rows to the validation and training sets until we reach the desired size for validation
    for xx in image_dict:
        xx_rows = image_dict[xx]
        val_size_for_xx = min(val_set_size - val_count, len(xx_rows))
        val_writer.writerows(xx_rows[:val_size_for_xx])
        train_writer.writerows(xx_rows[val_size_for_xx:])
        val_count += val_size_for_xx

        # If we have reached the desired validation size, break the loop
        if val_count >= val_set_size:
            break

    # Write any remaining rows to the training set
    for xx in image_dict:
        train_writer.writerows(xx_rows[val_size_for_xx:])

print(f"Total rows processed: {total_count}")
print(f"Rows in validation set: {val_count}")
print(f"Rows in training set: {total_count - val_count}")