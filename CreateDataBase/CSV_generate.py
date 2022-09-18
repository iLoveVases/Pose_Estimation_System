# IMPORTS ----------------------------------------------------------------------------
import numpy as np
import os
from definitions import LEARNING_DATA_PATH, CLASSES, CSV_PATH, CSV_NAME

# VARIABLES ----------------------------------------------------------------------------
csv_file = CSV_PATH + rf"/{CSV_NAME}.csv"

label = "nose, nose, left eye, left eye, right eye, right eye, left ear, left ear, right ear, right ear, " \
        "left shoulder, left shoulder, right shoulder, right shoulder, left elbow, left elbow, right elbow, " \
        "right elbow, left wrist, left wrist, right wrist, right wrist, left hip, left hip, right hip, right hip, " \
        "left knee, left knee, right knee, right knee, left ankle, left ankle, right ankle, right ankle, POSE"

# FUNCTIONS ----------------------------------------------------------------------------


def count(path_to_count):
    path, dirs, files = next(os.walk(path_to_count + f"/"))
    file_count = len(files) - 1
    print(f"Got {file_count} npy files to work with")
    return file_count


# MAIN  ----------------------------------------------------------------------------

def write_csv():
    for position_csv in CLASSES:
        file = 0
        data_count = count(LEARNING_DATA_PATH + fr"\{position_csv}_npy/")

        for sample in range(1, data_count + 1):
            # loading data
            file += 1
            standing_path = LEARNING_DATA_PATH + fr"\{position_csv}_npy\{file}.npy"

            X = np.load(standing_path)  # X[point][coordinate]
            # reshaping from (1, 1, 17, 3) to (17, 3), cutting accuracy and receiving (17, 2), then flattening to (1D array)
            X = np.squeeze(X)
            X = X[:, :2]
            X = X.flatten()
            X = X.reshape((1, -1))  # (1,34)

            # adding POSE value
            if position_csv == 'standing':
                X = np.append(arr=X[0], values=0)
            elif position_csv == 'sitting':
                X = np.append(arr=X[0], values=1)
            elif position_csv == 'lying':
                X = np.append(arr=X[0], values=2)

            # reshaping after adding POSE
            X = X.reshape((1, -1))
            print(file, X)
            # collecting data
            if position_csv == 'standing' and file == 1:
                data = X
            else:
                data = np.append(data, X, axis = 0)

    np.savetxt(csv_file, data, fmt='%1.8f', delimiter=",", header=label, comments=' ')
    # fmt='%1.8f' -> 8 precision digits

