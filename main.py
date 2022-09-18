import CreateDataBase.Renamer as Renamer
from tkinter import *

if __name__ == '__main__':
    root = Tk(className=r"Pose Estimation System")
    root.geometry("350x150")

    # Define buttons and labels
    rename_label = Label(root, text="Prepare photo samples for processing")
    rename_button = Button(root, text="Rename", command=Renamer.arrange)

    collect_label = Label(root, text="Image Data Acquisition")
    collect_button = Button(root, text=" Collect ")  # Connect Function later

    convert_label = Label(root, text="Generate CSV file based on numerical data")
    convert_button = Button(root, text="Convert")  # Connect Function later

    # Positioning on the screen
    rename_label.place(x=100, y=23)
    rename_button.place(x=40, y=20)

    collect_label.place(x=100, y=53)
    collect_button.place(x=40, y=50)

    convert_label.place(x=100, y=83)
    convert_button.place(x=40, y=80)

    # Plot
    root.mainloop()
