import CreateDataBase.Renamer as Renamer
from tkinter import *

if __name__ == '__main__':
    root = Tk(className=r"Pose Estimation System")
    root.geometry("350x100")

    # Define buttons and labels
    rename_label = Label(root, text="Prepare photo samples for processing")
    rename_button = Button(root, text="Rename", command=Renamer.arrange)

    # Positioning on the screen
    rename_button.place(x=40, y=20)
    rename_label.place(x=100, y=23)

    # Plot
    root.mainloop()
