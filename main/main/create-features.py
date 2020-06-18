import os
import sys
import math

INFINITY = math.pow(10, 7)

CLEAN = "CLEAN"
NOISY = "NOISY"
NONE = "NONE"
NEG = "NEG"

PL = "PL"
NPL = "NPL"

FORMULA = "formula"

def sbflFeatures():

    final_data = {}
    item_data = []

    print("Creating from folder", sys.argv)

    files = os.listdir(sys.argv[1])
    
    for f in files:
        if (not "type" in f and not "ER5b" in f and not "ER5a" in f):
            list = final_data.get(FORMULA, [])
            list.append(f)
            final_data[FORMULA] = (list)
            
            print(f)

            for s in open(os.path.join(sys.argv[1], f), "r").readlines():
                function = s.split(" ")[0].strip()
                sus = s.split(" ")[1].strip()
                
                if sus == math.inf or sus == "Infinity":
                    sus = str(INFINITY)

                list = final_data.get(function, [])
                list.append(sus)
                final_data[function] = list
            print("-------------")

    for k in final_data.keys():
        temp = [].append(k)
        #print(",".join([k, ",".join(final_data[k]), "\n"]))

    classes = classFeatures(final_data.keys())
    classFeatsFile = open(sys.argv[18], "w")

    for c in classes.keys():
        classFeatsFile.write(",".join([c, str(classes[c])]))
        classFeatsFile.write("\n")

    temp = {}

    temp["simfix"] = profl_Simfix_Features(sys.argv[3], final_data.keys())
    
    temp["tbar"] = profl_TBarFam_Features(sys.argv[4], final_data.keys())
    temp["avatar"] = profl_TBarFam_Features(sys.argv[5], final_data.keys())
    temp["kpar"] = profl_TBarFam_Features(sys.argv[6], final_data.keys())
    temp["fixminer"] = profl_TBarFam_Features(sys.argv[7], final_data.keys())

    temp["arja"] =  profl_ArjaFam_Features(sys.argv[8], final_data.keys())
    temp["genprog"]  =  profl_ArjaFam_Features(sys.argv[9], final_data.keys())
    temp["kali"]  =  profl_ArjaFam_Features(sys.argv[10], final_data.keys())
    temp["rsrepair"]  =  profl_ArjaFam_Features(sys.argv[11], final_data.keys())

    temp["jgenprog"]  = profl_jGenProgFam_Features(sys.argv[12], final_data.keys())
    temp["jkali"]  = profl_jGenProgFam_Features(sys.argv[13], final_data.keys())
    temp["jmutrepair"]  = profl_jGenProgFam_Features(sys.argv[14], final_data.keys())
    temp["cardumen"]  = profl_jGenProgFam_Features(sys.argv[15], final_data.keys())

    temp["acs"] = profl_acs_Features(sys.argv[16], final_data.keys())
    temp["dynamoth"]  = profl_dynamoth_Features(sys.argv[17], final_data.keys())

    output = {}

    for k in temp.keys():
        for j in temp[k].keys():
            for i in temp[k][j]:
                list = output.get(j, [])
                if(j is FORMULA):
                    list.append("-".join([k,i]))
                else:
                    list.append(str(temp[k][j][i]))
                output[j] = list
    
    numFeatsFile = open(sys.argv[19], "w")

    for m in output.keys():
        numFeatsFile.write(",".join([m, ",".join(output[m]), ",".join(final_data[m])]))
        numFeatsFile.write("\n")

def classFeatures(keys):
    class_data = {}
    data = open(sys.argv[2], "r").readlines()

    for k in keys:
        value = 0
        for d in data:
            if k in d.replace(":","."):
                value = 1

        class_data[k] = value
    return class_data

def profl_dynamoth_Features(file, keys):
    data = newDataDict(keys)

    plausible = False
    modifiedMethod = ""
    fixType = None

    try:
        for m in open(file, "r").readlines():
            m = m.strip()
        
            if "Test suite results" in m:
                if "ff=0" in m and "pf=0" in m:
                    plausible =  True
                else:
                    plausible = False
            elif "modifedMethodSignature" in m:
                modifiedMethod = m.split("=")[1].replace(":", ".").strip()
            elif "Fix detected" in m:
                if "CleanFix" in m:
                    fixType = CLEAN
                elif "NoisyFix" in m:
                    fixType = NOISY
                elif "NoneFix" in m:
                    fixType = NONE
                elif "NegFix" in m:
                    fixType = NEG

                try:
                    data[modifiedMethod][fixType] = data[modifiedMethod].get(fixType, 0) + 1
            
                    if plausible:
                        data[modifiedMethod][PL] = data[modifiedMethod].get(PL, 0) + 1
                    else:
                        data[modifiedMethod][NPL] = data[modifiedMethod].get(NPL, 0) + 1
                except:
                    pass

                modifiedMethod = ""
                fixType = None
                plausible = False
    except:
        data = newDataDict(keys)
    return data
def profl_acs_Features(file, keys):
    data = newDataDict(keys)

    plausible = False
    modifiedMethod = ""
    fixType = None

    try:
        for m in open(file, "r").readlines():
            m = m.strip()

            if "ff=" in m and "pf=" in m:
                if "ff=0" in m and "pf=0" in m:
                    plausible = True
                else:
                    plausible = False
            elif "Buggy method =" in m:
                modifiedMethod = m.split(" = ")[1].replace(":", ".").strip()
            elif "Fix detected" in m:
                if "CleanFix" in m:
                    fixType = CLEAN
                elif "NoisyFix" in m:
                    fixType = NOISY
                elif "NoneFix" in m:
                    fixType = NONE
                elif "NegFix" in m:
                    fixType = NEG

                try:
                    data[modifiedMethod][fixType] = data[modifiedMethod].get(fixType, 0) + 1
            
                    if plausible:
                        data[modifiedMethod][PL] = data[modifiedMethod].get(PL, 0) + 1
                    else:
                        data[modifiedMethod][NPL] = data[modifiedMethod].get(NPL, 0) + 1
                except:
                    pass
                modifiedMethod = ""
                fixType = None
                plausible = False
    except:
        data = newDataDict(keys)
    return data

def profl_jGenProgFam_Features(file, keys):
    data = newDataDict(keys)

    plausible = False
    modifiedMethod = ""
    fixType = None

    try:
        for m in open(file, "r").readlines():
            m = m.strip()

            if "Regression testing results" in m:
                if "ff=0" in m and "pf=0" in m:
                    plausible = True
                else:
                    plausible = False
            elif "Buggy code located" in m:
                modifiedMethod = m.split(" ")[-1][1:-1].replace(":", ".")
            elif "Fix detected" in m:
                if "CleanFix" in m:
                    fixType = CLEAN
                elif "NoisyFix" in m:
                    fixType = NOISY
                elif "NoneFix" in m:
                    fixType = NONE
                elif "NegFix" in m:
                    fixType = NEG

                try:
                    data[modifiedMethod][fixType] = data[modifiedMethod].get(fixType, 0) + 1
            
                    if plausible:
                        data[modifiedMethod][PL] = data[modifiedMethod].get(PL, 0) + 1
                    else:
                        data[modifiedMethod][NPL] = data[modifiedMethod].get(NPL, 0) + 1

                except:
                    pass
                modifiedMethod = ""
                fixType = None
                plausible = False
    except:
        data = newDataDict(keys)
    return data

def profl_ArjaFam_Features(file, keys):
    data = newDataDict(keys)

    plausible = False
    modifiedMethod = ""

    try:
        for m in open(file, "r").readlines():
            m = m.strip()

            if "Number of failed tests" in m:
                if "Number of failed tests: 0" in m:
                    plausible = True
                else:
                    plausible = False
            elif "Modified method" in m:
                modifiedMethod = m.split("Modified method ")[1].split(" at ")[0]
                modifiedMethod = modifiedMethod.replace(":", ".")
            elif "PatchCategory" in m:
                try:
                    if plausible:
                        data[modifiedMethod][PL] = data[modifiedMethod].get(PL, 0) + 1
                    else:
                        data[modifiedMethod][NPL] = data[modifiedMethod].get(NPL, 0) + 1

                    if "CleanFix" in m:
                        data[modifiedMethod][CLEAN] = data[modifiedMethod].get(CLEAN, 0) + 1
                    elif "NoisyFix" in m:
                        data[modifiedMethod][NOISY] = data[modifiedMethod].get(NOISY, 0) + 1
                    elif "NoneFix" in m:
                        data[modifiedMethod][NONE] = data[modifiedMethod].get(NONE, 0) + 1
                    elif "NegFix" in m:
                        data[modifiedMethod][NEG] = data[modifiedMethod].get(NEG, 0) + 1
                except:
                    pass
                modifiedMethod = ""
                plausible = False
    except:
        data = newDataDict(keys)
    return data

def profl_TBarFam_Features(file, keys):
    data = newDataDict(keys)
    
    plausible = False
    fixType = ""
    try:
        for m in open(file, "r").readlines():
            m = m.strip()

            if "Testing New Patch" in m:
                plausible = False
                fixType = ""
            elif "Patch test results" in m:
                if "ff=0" in m and "pf=0" in m:
                    plausible = True
                else:
                    plausible = False
            elif "Fix found" in m:

                if "CleanFix" in m:
                    fixType = CLEAN
                elif "NoisyFix" in m:
                    fixType = NOISY
                elif "NoneFix" in m:
                    fixType = NONE
                elif "NegFix" in m:
                    fixType = NEG

            elif "Mutated = " in m and (not fixType is ""):
                method = m.split("Mutated = ")[1].split(" in ")[0].replace(":", ".").strip()

                try:
                    data[method][fixType] = data[method].get(fixType, 0) + 1
            
                    if plausible:
                        data[method][PL] = data[method].get(PL, 0) + 1
                    else:
                        data[method][NPL] = data[method].get(NPL, 0) + 1
                except:
                    pass
                method = ""
                fixType = None
                plausible = False
    except:
        data = newDataDict(keys)
    return data
  
def profl_Simfix_Features(file, keys):
    data = newDataDict(keys)

    lastMethodModified = ""
    try:
        for m in open(file, "r").readlines():
            m = m.strip()

            if "Method modified" in m:
                lastMethodModified = m.split("Method modified = ")[1].strip()
                lastMethodModified = lastMethodModified.replace(":", ".")
            elif "CleanFix" in m:
                try:
                    data[lastMethodModified][CLEAN] = data[lastMethodModified].get(CLEAN, 0) + 1 
                    data[lastMethodModified][PL] = data[lastMethodModified].get(PL, 0) + 1 
                except:
                    pass
            elif "Fix detected" in m:
                try:

                    data[lastMethodModified][NPL] = data[lastMethodModified].get(NPL, 0) + 1 

                    if "NoisyFix" in m:
                        data[lastMethodModified][NOISY] = data[lastMethodModified].get(NOISY, 0) + 1 
                    elif "NoneFix" in m:
                        data[lastMethodModified][NONE] = data[lastMethodModified].get(NONE, 0) + 1 
                    elif "NegFix" in m:
                        data[lastMethodModified][NEG] = data[lastMethodModified].get(NEG, 0) + 1 
                except:
                    pass
    except:
            data = newDataDict(keys)
    return data

def newDataDict(keys):
    data = {}

    for k in keys:
        data[k] = {}

        data[k][CLEAN] = 0
        data[k][NOISY] = 0
        data[k][NONE] = 0
        data[k][NEG] = 0

        data[k][PL] = 0
        data[k][NPL] = 0
    return data

sbflFeatures()