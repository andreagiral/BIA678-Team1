#Component 1: Set Up
##1a: Set up a virtual enviroment

##1b: In the terminal insert the following: pip install pandas requests sqlalchemy numpy scipy sklearn matplotlib prophet xgboost

##1c: Run the following code to determine if any of the packages were not installed propertly

modules = [
    "pandas", "requests", "sqlalchemy", "numpy", "scipy",
    "sklearn", "matplotlib", "prophet", "xgboost"
]

for module in modules:
    try:
        __import__(module)
        print(f"{module} is installed")
    except ImportError:
        print(f"{module} is not installed")
