import os
import time

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
  
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")

try:
    from src.repop_algorithm import RepopAlgorithm, read_config_file
except:
    from repop_algorithm import RepopAlgorithm, read_config_file


input_file = read_config_file('scripts/input_example.yml')
outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


model = RepopAlgorithm(input_file)


model.run('outputs/test_' + outtime,
          # No conf given runs all listed configs
          # configuration=['dmo_fragile']
          )

