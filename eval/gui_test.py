from tkinter import *


class TestGUI(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.currentOutput = []

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.gridFrame = Frame(self)
        self.gridFrame.grid(row=2, column=1, sticky=W+E+N+S)

        self.text = Text(self.gridFrame, wrap="none", height=1, width=80, bd=3, font=("Helvetica", "14"))
        self.text.bind("<Return>", self.retrieve_input)
        self.text.grid(row=0, column=0, sticky=W+E)

        self.wordProposal = Listbox(self.gridFrame, bd=3, font=("Helvetica", "12"), height=3)
        self.wordProposal.bind('<<ListboxSelect>>', self.select_word)
        self.wordProposal.grid(row=1, column=0, sticky=W + E)

    def retrieve_input(self, evt):
        inputValue = self.text.get("1.0", "end-1c")
        if len(inputValue) > 30:
            inputValue = inputValue[len(inputValue) - 30:]
        print(inputValue)
        self.currentOutput = ["hallo", "test"]
        for index in range(len(self.currentOutput)):
            self.wordProposal.insert(index, self.currentOutput[index])

        return "break"

    def select_word(self, evt):
        w = evt.widget
        sentence = self.text.get("1.0", "1.end")

        if len(w.curselection()) > 0:
            prefix = " "
            if len(sentence) > 0:
                last_char = sentence[len(sentence)-1]
                found_whitespace = re.search('\s', last_char)
                if found_whitespace:
                    prefix = ""

            index = int(list(w.curselection())[0])
            value = w.get(index)

            self.text.insert(END, prefix + value + " ")
            self.wordProposal.delete(0, len(self.currentOutput))
            self.currentOutput = []

if __name__ == "__main__":
    root = Tk()
    TestGUI(root).pack(fill="both", expand=True)
    root.mainloop()
