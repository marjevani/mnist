from tkinter import *
import threading
from tkinter import messagebox

ROWS = 20
COLS = 20

# Paint module has to get his manager in constructor.
# manager has to implement:
    # 1)  "pre_proccess" function for pre processing the img before evaluate.
    # 2) "send_eval" for evalutaing the img data
    #    this function can update Paint status bar with the result by using "paint.set_status" function.
    #    NOTICE - you may wants to evaluate in different thread to prevent GUI freezing

class Paint(object):
    def __init__(self, im):
        self.lock = threading.RLock()
        self.manager = im

        ## build GUI ##
        self.root = Tk()

        # Create a grid of None to store the references to the tiles
        self.tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

        self.canvas = Canvas(self.root, bg='white', width=600, height=600)
        self.canvas.grid(row=1, columnspan=5)
        self.canvas.bind("<B1-Motion>", self.callback)

        # create buttons
        self.clear_btn = Button(self.root, text='clear', command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0)

        self.send_btn = Button(self.root, text='send', command=self.send_eval)
        self.send_btn.grid(row=0, column=1)

        # Create eraser/painter radio buttons:
        erase_frame = Frame(self.root)
        self.eraser_on = BooleanVar()
        self.eraser_on.set(False)
        R1 = Radiobutton(erase_frame, text="painter", variable=self.eraser_on, value=False)
        R2 = Radiobutton(erase_frame, text="eraser", variable=self.eraser_on, value=True)
        erase_frame.grid(row=0, column=4)
        R1.pack(side="left")
        R2.pack(side="right")

        # boolean lock. logic - can't paint after send img & before clear canvas
        self.can_paint = True

        # Add status bar
        self.status_bar = Label(self.root, text="Paint your digit", bd=1, relief=SUNKEN, anchor=W,font=('TkDefaultFont', 18))
        self.status_bar.grid(column=0, row=2, columnspan=5, sticky='we')

        self.root.mainloop()

    def callback(self,event):
        # Get rectangle diameters
        col_width = int(self.canvas.winfo_width() / COLS)
        row_height = int(self.canvas.winfo_height() / ROWS)
        # Calculate column and row number
        col = event.x//col_width
        row = event.y//row_height
        if event.x < 0 or event.x >= self.canvas.winfo_width():
            return
        if event.y < 0 or event.y >= self.canvas.winfo_height():
            return

        # paint or erase rectangle
        if not self.can_paint:
            self.set_status("ERROR - please clear before painting!!!")
        elif not self.eraser_on.get():
                # If the tile is not filled, create a rectangle
                self.create_rec(col, row)
                self.create_rec(col+1, row)
                self.create_rec(col, row+1)
                self.create_rec(col+1, row+1)
        else:
            self.delet_cel(col, row )
            self.delet_cel(col, row + 1)
            self.delet_cel(col + 1, row )
            self.delet_cel(col + 1, row + 1)

    def delet_cel(self, col, row):
        if (col < COLS) and (row < ROWS) and self.tiles[row][col] is not None:
            self.canvas.delete(self.tiles[row][col])
            self.tiles[row][col] = None

    def create_rec(self, col, row):
        colorval = "#%02x%02x%02x" % (0, 0, 0)
        # Get rectangle diameters
        col_width = int(self.canvas.winfo_width() / COLS)
        row_height = int(self.canvas.winfo_height() / ROWS)

        if (col < COLS) and (row < ROWS) and self.tiles[row][col] is None:
            self.tiles[row][col] = self.canvas.create_rectangle(col * col_width, row * row_height,
                                                                        (col + 1) * col_width, (row + 1) * row_height,
                                                                        fill=colorval, outline=colorval)

    def send_eval(self):
        # change and check status
        eval_string = "Evaluating your paint now.."
        if(self.status_bar['text'] == eval_string ):
            # allready Evaluating - stop
            return
        self.can_paint = False
        self.set_status(eval_string)

        processed_img = self.manager.pre_process(self.tiles)
        self.show_img(processed_img)
        self.manager.send_eval(self)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.can_paint = True
        self.set_status("Paint your digit")

    def show_img(self, img):
        self.canvas.delete("all")
        # paint img on canvas
        col_width = int(self.canvas.winfo_width() / 28)
        row_height = int(self.canvas.winfo_height() / 28)

        centerd = [[None] * 28] * 28
        for row in range(28):
            for col in range(28):
                num = img[row][col]
                colorval = "#%02x%02x%02x" % (num, num, num)
                centerd[row][col] = self.canvas.create_rectangle(col * col_width, row * row_height,
                                                                 (col + 1) * col_width,
                                                                 (row + 1) * row_height, fill=colorval,
                                                                 outline=colorval)

    # this function calls from outside
    # those, need to be synchronized
    def set_status(self, msg=''):
        # synchronized function
        self.lock.acquire()
        self.status_bar.configure(text=msg)
        self.lock.release()

    def pop_up(self, title, msg):
        messagebox.showinfo(title, msg)