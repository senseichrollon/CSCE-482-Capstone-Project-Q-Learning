import tkinter as tk
import subprocess
from tkinter import ttk
from googletrans import Translator
from googletrans import LANGUAGES

language_dict = {name: code for code, name in LANGUAGES.items()}
def translate_labels(root, dest_language):
    translator = Translator()
    print(dest_language)
    dest_language_code = language_dict[dest_language]

    for widget in root.winfo_children():
        if isinstance(widget, (tk.Label, tk.Button)):
            original_text = widget.cget("text")
            translated_text = translator.translate(original_text, dest=dest_language_code).text
            widget.config(text=translated_text)


def run_backend():
    arg1 = entry1.get()
    arg2 = entry2.get()
    arg3 = entry3.get()
    arg4 = entry4.get()
    arg5 = entry5.get()
    subprocess.run(['python', 'test_backend.py', '--version', arg1, '--operation', arg2, '--save-path', arg3, '--reward-function', arg4, '--map', arg5])


root = tk.Tk()
root.title("CARLA User Interface")

style = ttk.Style(root)
style.theme_use('classic')

# Create input fields
label1 = tk.Label(root, text="Version:")
label1.grid(row=0, column=0, padx=5, pady=5)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=5, pady=5)

label2 = tk.Label(root, text="Operation:")
label2.grid(row=1, column=0, padx=5, pady=5)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=5, pady=5)

label3 = tk.Label(root, text="Save Path:")
label3.grid(row=2, column=0, padx=5, pady=5)
entry3 = tk.Entry(root)
entry3.grid(row=2, column=1, padx=5, pady=5)

label4 = tk.Label(root, text="Reward Function:")
label4.grid(row=3, column=0, padx=5, pady=5)
entry4 = tk.Entry(root)
entry4.grid(row=3, column=1, padx=5, pady=5)

label5 = tk.Label(root, text="Map:")
label5.grid(row=4, column=0, padx=5, pady=5)
entry5 = tk.Entry(root)
entry5.grid(row=4, column=1, padx=5, pady=5)

# Create 'Run' button
run_button = tk.Button(root, text="Run Backend", command=run_backend)
run_button.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

# Create language dropdown menu
languages = list(LANGUAGES.values())
language_select_label = tk.Label(root, text="Select Language:")
language_select_label.grid(row=5, column=0, padx=5, pady=5)
#languages = [ 'fr', 'en', 'sp']  # Add more languages as needed
language_select = ttk.Combobox(root, values=languages)

language_select.grid(row=5, column=1, padx=5, pady=5)


language_select.current(0)  # Set default language
language = language_select.current(0)  # Set default language
#root.language = language_select.get()
# Create 'Translate' button
translate_button = tk.Button(root, text="Translate", command=lambda:translate_labels(root, language_select.get()))
translate_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)


# Start the GUI event loop
root.mainloop()