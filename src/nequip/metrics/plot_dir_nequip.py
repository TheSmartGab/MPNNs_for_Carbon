import sys, os

data_filename = "metrics.csv"
scripts_dir = os.path.dirname(os.path.abspath(__file__))
plot_script = os.path.join(scripts_dir, "plot_nequip.py")

if __name__ == "__main__":
    """
    This script looks for all metrics.csv in all subdirectories of given target, and calls plot_nequip.py on them to do the plotting and printing to json the test results. all results are stored in the same directory as the metrics.csv files.
    csv files should be in a directory without subdirectories.
    If this should no longer be the case, change the logic.
    """
    if len(sys.argv) < 1:
        target = input("target directory: ")
    else:
        target = sys.argv[1]

    walk = os.walk(target)
    for (root,dir,files) in walk:
        if not dir: # take the last part of the walk where there are no subdirectories and only the 
            if data_filename in files:
                file = os.path.join(root, data_filename)
                os.system(f"python3 {plot_script} {file}")
        




    