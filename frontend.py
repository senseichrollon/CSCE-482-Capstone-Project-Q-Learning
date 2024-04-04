import tkinter as tk
import subprocess

def run_backend():
    arg1 = entry1.get()
    arg2 = entry2.get()
    subprocess.run(['python', 'bak.py', '--operation', arg1, '--reward-function', arg2])


root = tk.Tk()
root.title("Frontend GUI")

# Create input fields
label1 = tk.Label(root, text="Argument 1:")
label1.grid(row=0, column=0, padx=5, pady=5)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=5, pady=5)

label2 = tk.Label(root, text="Argument 2:")
label2.grid(row=1, column=0, padx=5, pady=5)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=5, pady=5)

# Create 'Run' button
run_button = tk.Button(root, text="Run Backend", command=run_backend)
run_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)


# Start the GUI event loop
root.mainloop()