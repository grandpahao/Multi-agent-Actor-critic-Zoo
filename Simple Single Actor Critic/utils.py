import os


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
