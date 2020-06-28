import os

def main():
    dir = "D:\\Dropbox\\Project Repair Files\\Machine Learning"
    newFeats_dir = "C:\\Users\\prodi\\Downloads\\multi_patch_group.tar\\multi_patch_group\\"

    for f in os.listdir(dir):
        parts = f.split(".")[0].split("-")

        if f.endswith(".numFeats") and not "Math-66" in f:
            originalFile = open(os.path.join(dir, f), "r")
            originalData = originalFile.readlines()[1:]
            originalData = replace(originalData)

            try:
                newFile = open(newFeats_dir + parts[0] + "\\" + parts[1] + ".txt", "r")
                newData = newFile.readlines()
                newData = replace(newData)

                for line in newData:
                    line_data = line.strip().split(" ")

                    insert_data = []
                    PL = 0
                    NPL = 0
                    for cat in reversed(line_data[1:]):
                        cat_data = cat.split("*")
                        
                        if "Clean" in cat_data[0]:
                            PL = PL + float(cat_data[1])
                        else:
                            NPL = PL + float(cat_data[1])

                        insert_data.append(float(cat_data[1]))

                    insert_data.append(PL)
                    insert_data.append(NPL)
                    
                    find(line_data[0], originalData, insert_data)

                    updatedData = open(os.path.join(dir, f), "w")

                    for data in originalData:
                        updatedData.write(data.strip())
                        updatedData.write("\n")
                
            except Exception as e:
                processMissingFile(originalData)

                updatedData = open(os.path.join(dir, f), "w")

                for data in originalData:
                    updatedData.write(data.strip())
                    updatedData.write("\n")

def find(query, array, cat_data):

    result = []

    for i in range(len(array)):
        if query in array[i]:
            profl_data = array[i].split(",")[:91]
            sbfl_data = array[i].split(",")[91:]

            result.extend(profl_data)
            result.extend(cat_data)
            result.extend(sbfl_data)

            temp = []

            for index in result:
                if isinstance(index, str):
                    temp.append(index) 
                else:
                    temp.append(str(index))

            array[i] = ",".join(temp)            
            
def replace(array):

    data = []

    for a in array:
        data.append(a.replace(":", "."))

    return data

def processMissingFile(data):

    for i in range(len(data)):
        result = []

        profl_data = data[i].split(",")[:91]
        sbfl_data = data[i].split(",")[91:]

        result.extend(profl_data)
        result.extend([0.0,0.0,0.0,0.0,0.0,0.0])
        result.extend(sbfl_data)

        temp = []

        for index in result:
            if isinstance(index, str):
                temp.append(index) 
            else:
                temp.append(str(index))
        
        data[i] = ",".join(temp)   
        
main()
