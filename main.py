import CreateDataBase.Renamer as Renamer
import CreateDataBase.Collecting_Data as Collecting_Data
import CreateDataBase.CSV_generate as CSV_generate
from tkinter import *

if __name__ == '__main__':
    root = Tk(className=r"Pose Estimation System")
    root.geometry("350x150")


    # Define buttons and labels
    rename_label = Label(root, text="Prepare photo samples for processing")
    rename_button = Button(root, text="Rename", command=Renamer.arrange)

    collect_label = Label(root, text="Image Data Acquisition")
    collect_button = Button(root, text=" Collect ", command=Collecting_Data.gather_data)

    convert_label = Label(root, text="Generate CSV file based on numerical data")
    convert_button = Button(root, text="Convert", command=CSV_generate.write_csv)

    csv_name = Entry(root, width=15)
    csv_name_label = Label(root, text="CSV Name:")

    # Positioning on the screen

    rename_label.place(x=100, y=23)
    rename_button.place(x=40, y=20)

    collect_label.place(x=100, y=53)
    collect_button.place(x=40, y=50)

    convert_label.place(x=100, y=83)
    convert_button.place(x=40, y=80)

    csv_name.place(x=170, y=112)
    csv_name_label.place(x=100, y=110)

    # Plot
    root.mainloop()
