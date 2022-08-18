# IMPORTS ----------------------------------------------------------------------------
import glob
import os
from natsort import natsorted, ns
from definitions import IMAGES_PATHS
# VARIABLES ----------------------------------------------------------------------------
PATHS_TO_RENAME = [x+r'\\' for x in IMAGES_PATHS]

# MAIN  ----------------------------------------------------------------------------
for path in PATHS_TO_RENAME:
    # search for .jpg files
    pattern = path + "*.jpg"

    # List of the files that match the pattern
    result = glob.glob(pattern)
    result = natsorted(result, alg=ns.IGNORECASE)
    # Iterating the list with the count
    count = 1
    for file_name in result:
        old_name = file_name
        new_name = path + str(count) + "temp" + ".jpg"
        os.rename(old_name, new_name)
        count = count + 1

    result = glob.glob(pattern)
    result = natsorted(result, alg=ns.IGNORECASE)
    count = 1
    for file_name in result:
        old_name = file_name
        new_name = path + str(count) + ".jpg"
        os.rename(old_name, new_name)
        count = count + 1

    # printing all revenue txt files
    res = sorted(glob.glob(path + "*.jpg"))
    # for name in res:
        # print(name)