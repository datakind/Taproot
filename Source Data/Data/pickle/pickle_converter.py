def convert_xls_to_pickle(xls_path):
    '''
    Takes the raw xls files, loads them into Pandas dataframe and exports pickle objects
    '''
    import pandas as pd
    
    files = ["Projects Application-Project Evaluation links.xlsx",\
            "Projects Applications.xlsx", \
            "Projects Awarded.xlsx", \
            "Projects Evaluations (2010 - Q2).xls", \
            "Projects-Volunteers Links.xlsx", \
            "Volunteers Applications.xlsx", \
            "Volunteers Evaluation (2010 - Q2).xlsx"]
    
    for file in files:
        file_name = file.split(".")[0]
        df = pd.read_excel((xls_path + file))
        df.to_pickle(file_name + ".p")
        print("Finished file {0}".format(file))
        
        
    return
        
        
convert_xls_to_pickle(xls_path = r'../')