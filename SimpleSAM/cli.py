import argparse
import os

from SimpleSAM.gui import GUI

def main():
    arg_parser = argparse.ArgumentParser(description='A Simple GUI for making labeled COCO datasets using Meta\'s Segment Anything Model')

    args = arg_parser.parse_args()
    gui = GUI()
    gui.run()


if __name__ == "__main__":
    main()