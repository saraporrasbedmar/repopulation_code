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

# def save_example_with_callable(Vmax, params):
#     return (0.5 * (u.km/ u.s) + 0.01 * Vmax)**params


# input_file['repopulations']['params_to_save']['ex_with_callable'] = {
#     'formula': save_example_with_callable,
#     'params': 10, 'variables': 'Vmax'}


# Careful with this, because this technically works, but the SRD
# does NOT depend on Vmax. Rather, Vmax is taken as the variable
# necessary to calculate the probability distribution function of the
# SRD. Similar to the SHVF inputs. The rest of the functions do
# introduce the variables.
# def srd_example(Vmax, params):
#     return (0.5 + 0.01 * Vmax**params)*(Vmax > 150)

# input_file['configurations']['dmo_fragile2']['SRD'] = {
#     'formula': srd_example, 'params': 1e2}


model = RepopAlgorithm(input_file)


model.run('outputs/test_' + outtime,
          # No conf given runs all listed configs
          # configuration=['dmo_fragile']
          )

