import sys, os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def clearoutputs(output_filepath):
    is_file_path = os.path.isdir(output_filepath)
    if is_file_path:
        print("file path already exists - generating new path")
        while is_file_path:
            output_filepath = output_filepath+"_new"
            is_file_path = os.path.isdir(output_filepath)
    
    os.makedirs(f"{output_filepath}")
    
    #shutil.rmtree(f"{output_filepath}/")
    
    os.makedirs(f"{output_filepath}/wavefunctions")
    os.makedirs(f"{output_filepath}/innerprods")
    os.makedirs(f"{output_filepath}/densities")
    os.makedirs(f"{output_filepath}/debugging")
    os.makedirs(f"{output_filepath}/orbitals")
    os.makedirs(f"{output_filepath}/orbitals/hartree-fock")
    os.makedirs(f"{output_filepath}/states")
    f = open(f"{output_filepath}/energies.txt", "x")

    
    print(f"working directory is now {output_filepath}")
    return output_filepath

