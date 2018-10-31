"""
This module defines the printing functions

author: Louis-Émile Robitaille (l3robot)
date-of-creation: 2018-10-30
"""

ERROR, WARNING, INFO = list(range(3))
HEADER = {
    ERROR: " [!] Error: ",
    WARNING: " [?] Warning: ",
    INFO: " [-] Info: ",
}


def printl(lvl, string, file=None):
    """
    Prints with a level of message to std output or a file

    Args:
        lvl (int): level of message
        string (str): string to write
        file (file-like object): the file to write (optional)
    """
    final_string = f'{HEADER}{string}'
    if file is None:
        print(final_string)
    else:
        f.write(final_string)
