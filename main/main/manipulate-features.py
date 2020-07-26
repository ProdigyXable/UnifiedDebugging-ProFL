
import os

def main():
    dir = "D:\\Dropbox\\Project Repair Files\\Machine Learning"


    for f in os.listdir(dir):
        parts = f.split(".")[0].split("-")

        if f.endswith(".numFeats") or f.endswith(".classFeats"):
            originalFile = open(os.path.join(dir, f), "r")
            originalData = originalFile.readlines()[1:]
            newFile = open(os.path.join(dir, f + ".xia"), "w")

            for line in originalData:
                originalDataSplit = line.split(",")
                sig = originalDataSplit[0]
                other = originalDataSplit[1:]

                message = sig + " " + ",".join(other)
                print(message)

                newFile.write(message)        
main()
