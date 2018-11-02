DEBUG = True
# DEBUG = False

def debug_print(*str, end= "\n"):
    if DEBUG:
        print(str, end=end)