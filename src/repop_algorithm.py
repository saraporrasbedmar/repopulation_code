import os
import yaml
import copy
import time
import h5py
import psutil
import logging
import inspect

import numpy as np

from scipy.optimize import newton, minimize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_simpson, quad

from astropy import units as u
from astropy import constants as c


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    return mem


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


mass_energy2 = [
    (u.si.kg ** 2, u.si.J ** 2.,
     lambda x: (x * c.c.value ** 4), lambda x: (x / c.c.value ** 4)),
    (u.kg ** 2. / u.m ** 5, u.J ** 2. / u.m ** 5,
     lambda x: x * c.c ** 4, lambda x: x / c.c ** 4)
]


class RepopAlgorithm:
    def __init__(self, path_input):

        self._process_time = time.process_time()
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        self.path_output = None
        self.configuration = None

        if isinstance(path_input, str):
            self.input_dict = read_config_file(path_input)
        else:
            self.input_dict = path_input

        self.input_dict_strings = copy.deepcopy(self.input_dict)

        try:
            self.rng = np.random.default_rng(
                seed=self.input_dict['repopulations']['rng_seed'])

            print('Warning: seed for np.random has been explicitly set'
                  ' to: seed='
                  + str(self.input_dict['repopulations']['rng_seed'])
                  + '. Only use this setting for testing purposes.')
        except ValueError:
            self.rng = np.random.default_rng(seed=None)

        self._num_subs_max = int(1e6)

        self._number_highest = int(
            self.input_dict['repopulations']['number_highest'])

        self._its = self.input_dict['repopulations']['number_iterations']
        self._param_repopulation = self.input_dict['repopulations'][
            'param_to_repopulate']

        if self._param_repopulation == 'Vmax':
            self._concentr = 'Cv'
            self._calculate_from = 'Vmax_Cv'
        elif self._param_repopulation == 'Mass':
            self._concentr = 'Cmass'
            self._calculate_from = 'mass_Cmass'
        else:
            print(
                'Warning: parameter to repopulate is not recognized: '
                + self._param_repopulation
                + '\n    Most formulas will not work in this regime,'
                  ' such as use_spherical_shells, R_t, R_s, Roche,'
                  ' engulfs_Earth, Jfactor values.\n'
                  "\n    Values accepted for input_dict['repopulations']["
                    "'param_to_repopulate']: Mass, Vmax"
                  )
            self._concentr = 'concentration'

        try:
            self._prntfrq = int(
                self.input_dict['repopulations']['print_frequency'])
            if self._prntfrq <= 0:
                self._prntfrq = None
        except ValueError:
            self._prntfrq = None

        self.RangeMin = self.input_dict['repopulations']['RangeMin']
        self.RangeMax = self.input_dict['repopulations']['RangeMax']

        def value_with_unit(data_dict):
            if isinstance(data_dict, dict):
                for key, value in data_dict.items():
                    try:
                        if ('unit' in data_dict[key].keys()
                        and 'value' in data_dict[key].keys()):
                            data_dict[key] = (
                                data_dict[key]['value']
                                * u.Unit(data_dict[key]['unit']))
                    except (TypeError, AttributeError):
                        continue

                    if isinstance(value, dict):
                        value_with_unit(value)
                    elif isinstance(value, list):
                        for idx, item in enumerate(value):
                            if isinstance(item, dict):
                                value_with_unit(item)

            elif isinstance(data_dict, list):
                for idx, item in enumerate(data_dict):
                    if isinstance(item, dict):
                        value_with_unit(item)
            return data_dict

        value_with_unit(self.input_dict)

        self._dist_Earth_GC = np.sqrt(sum(
            self.input_dict['host']['position_Earth'] ** 2.))

        self._m_min = None
        self._m_max = None

        self.subhalo_data = {}

    def logging_info(self, text):
        if self.input_dict['repopulations']['verbose']:
            logging.info(
                '%.2fs: %s' % (time.process_time()
                               - self._process_time, text))
            self._process_time = time.process_time()

    def run(self, path_output, configuration=None):

        if configuration is None:
            config_list = self.input_dict['configurations'].keys()
        elif isinstance(configuration, str):
            config_list = [configuration]
        elif isinstance(configuration, list):
            config_list = configuration
        else:
            raise TypeError(
                'Configuration type not accepted.\n'
                + 'type: ' + type(configuration) + '\n'
                + 'configuration value: ' + configuration)

        self.path_output = path_output + '/'
        print(path_output)

        if not os.path.exists(path_output + '/'):
            os.makedirs(path_output + '/')

        for ii in config_list:
            self.configuration = ii
            self._param_repopulation = self.input_dict['repopulations'][
                'param_to_repopulate']
            aa = self.number_subhalos(
                self.RangeMin, self.RangeMax, force_no_fraction=True)
            print()
            print(self.configuration)
            print(self.RangeMin, self.RangeMax)
            print('    Max number of repop subhalos: %i' % aa)

            if self._number_highest > aa:
                print(
                    'Warning: number of requested subhalos to save is\n'
                    'higher than the total subhalos that are created.\n'
                    'Therefore, all subhalos will be saved.\n'
                    'Change this parameter if needed in:\n'
                    "input_dict['repopulations']['number_highest']")

            if self.input_dict['repopulations']['save_full_repop']:
                self.interior_full_repop()
            else:
                self.interior_brightest()

        # Save input data in a file in the outputs directory
        file_inputs = open(self.path_output + 'input_data.yml', 'w')
        aaa = copy.deepcopy(self.input_dict_strings)
        aaa = self.change_callables_into_strings(aaa)
        yaml.dump(aaa, file_inputs,
                  default_flow_style=False, allow_unicode=True)
        file_inputs.close()
        return

    def change_callables_into_strings(self, data_dict):
        if isinstance(data_dict, dict):
            for key, value in data_dict.items():
                if callable(value):
                    data_dict[key] = inspect.getsource(value).strip()
                elif isinstance(value, dict):
                    self.change_callables_into_strings(value)
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        if callable(item):
                            value[idx] = inspect.getsource(item).strip()
                        elif isinstance(item, dict):
                            self.change_callables_into_strings(item)

        elif isinstance(data_dict, list):
            for idx, item in enumerate(data_dict):
                if callable(item):
                    data_dict[idx] = inspect.getsource(item).strip()
                elif isinstance(item, dict):
                    self.change_callables_into_strings(item)
        return data_dict

    def calculate_formula(self, xx, formula, params=None):

        if isinstance(formula, str):

            if hasattr(self, formula):
                return getattr(self, formula)(xx, **params)

            bb = {}

            if isinstance(params, float) or isinstance(params, int):
                bb['params'] = params
            elif isinstance(params, list):
                bb['params'] = np.array(params, dtype=float)

            if isinstance(xx, list):
                xx = np.array(xx)
            bb['xx'] = xx

            # Evaluate formula
            try:
                aa = eval(formula, {}, bb)
            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula ' + formula
                    + f' with parameters {bb}: {e}')
            return aa

        elif callable(formula):
            if params is not None:
                if isinstance(params, dict):
                    return formula(xx, **params)
                return formula(xx, params)
            else:
                return formula(xx)

    def get_parameter(self, name, parametrization=None):
        """
        Retrieve parameter, computing if necessary.
        Uses values to avoid recomputation.
        """
        if name in self.subhalo_data:
            return self.subhalo_data[name]

        if hasattr(self, name):
            if isinstance(parametrization, dict):
                self.subhalo_data[name] = getattr(self, name)(
                    **parametrization)
                return self.subhalo_data[name]
            self.subhalo_data[name] = getattr(self, name)()
            return self.subhalo_data[name]

        formula = parametrization.get('formula', None)

        if isinstance(formula, str):

            if hasattr(self, formula):
                if isinstance(parametrization, dict):
                    try:
                        self.subhalo_data[name] = getattr(self, formula)(
                            **parametrization['params'])
                    except KeyError:
                        self.subhalo_data[name] = getattr(self, formula)()
                    return self.subhalo_data[name]
                self.subhalo_data[name] = getattr(self, formula)()
                return self.subhalo_data[name]

            bb = {}

            params = parametrization.get('params', None)
            if isinstance(params, float) or isinstance(params, int):
                bb['params'] = params
            elif isinstance(params, list):
                bb['params'] = np.array(params, dtype=float)

            variables = parametrization.get('variables', None)
            if isinstance(variables, str):
                bb[variables] = self.get_parameter(variables)
            elif isinstance(variables, list):
                for var in variables:
                    bb[var] = self.get_parameter(var)

            # Evaluate formula
            try:
                aa = eval(parametrization['formula'], {}, bb)
            except Exception as e:
                raise ValueError(
                    f'Error evaluating formula '
                    + parametrization['formula']
                    + f' with parameters {bb}: {e}')

            self.subhalo_data[name] = aa
            return self.subhalo_data[name]

        elif callable(formula):

            vars_for_func = {}
            variables = parametrization.get('variables', [])
            if isinstance(variables, str):
                variables = [variables]
            for var in variables:
                vars_for_func[var] = self.get_parameter(var)

            params = parametrization.get('params', None)
            if params is not None:
                vars_for_func['params'] = params

            self.subhalo_data[name] = formula(**vars_for_func)
            return self.subhalo_data[name]

    def calculate_characteristics_subhalo(
            self, xi_param=None, D_GC=None, position_Earth=None):

        self.subhalo_data = {}

        if xi_param is None:
            try:
                self._m_max = np.min((
                    newton(
                        self.xx, self._m_min * 1.05,
                        args=[self._m_min, self._num_subs_max]),
                    self.RangeMax
                ))
            except:
                self._m_max = self.RangeMax

            self._num_subhalos = self.number_subhalos(
                x_min=self._m_min, x_max=self._m_max,
                force_no_fraction=False)

            if self._num_subhalos <= 0:
                self.logging_info(
                    'No subhalos between %.2f and %.2f %s'
                      % (self._m_min, self._m_max,
                         self.input_dict['repopulations'][
                             'params_to_save'][
                             self._param_repopulation]['unit']
                         ))
                return

            self.logging_info(
                str((self._m_min, self._m_max, self._num_subhalos)))

            # Montecarlo algorithm for creating of subhalo Vmax
            x = np.geomspace(self._m_min, self._m_max, num=2000)
            y = self.calculate_formula(
                x,
                self.input_dict['configurations'][
                    self.configuration]['SHVF']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SHVF']['params']
            )

            cumul = cumulative_simpson(
                y=y * x * np.log(10), x=np.log10(x), initial=0)
            cumul /= cumul[-1]
            spline = UnivariateSpline(
                cumul, x, s=0, k=1, ext=1)

            self.subhalo_data[self._param_repopulation] = (
                    spline(self.rng.random(self._num_subhalos))
                    * u.Unit(
                self.input_dict['repopulations']['params_to_save'][
                    self._param_repopulation]['unit']))
        else:
            self.subhalo_data[self._param_repopulation] = xi_param

        if D_GC is None:
            # Montecarlo algorithm for creating of subhalo D_GC
            x = np.linspace(
                0. * u.kpc,
                (self.input_dict['host']['R_vir'].to(
                    u.Unit(self.input_dict
                           ['repopulations']['params_to_save']
                           ['D_GC']['unit']))),
                num=2000)

            if (self.input_dict['repopulations']['use_spherical_shells']
                    and self._Rcut + self._dist_Earth_GC
                    <= self.input_dict['host']['R_vir']):
                if self._Rcut < self._dist_Earth_GC:
                    x = np.linspace(
                        self._dist_Earth_GC - self._Rcut,
                        self._dist_Earth_GC + self._Rcut,
                        num=3000)
                else:
                    x = np.linspace(
                        0. * u.kpc,
                        self._dist_Earth_GC + self._Rcut,
                        num=3000)

            y = self.calculate_formula(
                x,
                self.input_dict['configurations'][
                    self.configuration]['SRD']['formula'],
                self.input_dict['configurations'][
                    self.configuration]['SRD']['params']
            )

            cumul = cumulative_simpson(y=y, x=x, initial=0)
            cumul /= cumul[-1]
            x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1
            spline = UnivariateSpline(
                cumul[x_min:], x[x_min:], s=0, k=1, ext=1)

            self.subhalo_data['D_GC'] = (
                    spline(self.rng.random(self._num_subhalos))
                    * u.Unit(self.input_dict
                             ['repopulations']['params_to_save']
                             ['D_GC']['unit']))
        else:
            self.subhalo_data['D_GC'] = D_GC

        if position_Earth is None:
            position_Earth = self.input_dict['host']['position_Earth']

        # Random distribution of subhalos around the celestial sphere
        self.subhalo_data['galactocentric_theta'] = self.rng.uniform(
            0, 2 * np.pi, self._num_subhalos) * u.rad

        self.subhalo_data['galactocentric_phi'] = np.arccos(
            2 * self.rng.uniform(0, 1, self._num_subhalos) - 1) * u.rad

        # Positions of the subhalos
        self.subhalo_data['galactocentric_X'] = (
                self.subhalo_data['D_GC']
                * np.cos(self.subhalo_data['galactocentric_theta'])
                * np.sin(self.subhalo_data['galactocentric_phi']))
        self.subhalo_data['galactocentric_Y'] = (
                self.subhalo_data['D_GC']
                * np.sin(self.subhalo_data['galactocentric_theta'])
                * np.sin(self.subhalo_data['galactocentric_phi']))
        self.subhalo_data['galactocentric_Z'] = (
                self.subhalo_data['D_GC']
                * np.cos(self.subhalo_data['galactocentric_phi']))

        self.subhalo_data['D_Earth'] = ((
            self.subhalo_data['galactocentric_X']
            - position_Earth[0]) ** 2
            + (self.subhalo_data['galactocentric_Y']
               - position_Earth[1]) ** 2
            + (self.subhalo_data['galactocentric_Z']
               - position_Earth[2]) ** 2
            ) ** 0.5

        self.get_parameter(self._concentr,
                parametrization=self.input_dict['configurations'][
                    self.configuration][self._concentr])

        self.logging_info('Subhalos positioned in the Galaxy.')

        for pn in self.input_dict['repopulations']['params_to_save']:
            self.logging_info('Calculating: %s' % pn)
            self._current_param_to_save = pn
            self.get_parameter(
                pn,
                self.input_dict['repopulations']['params_to_save'][pn])

        if self.input_dict['configurations'][
                    self.configuration]['use_Roche']:
            self.subhalo_data['survives_Roche'] = (
                    self.get_parameter('R_t')
                    > self.get_parameter('R_s')
            )

        self.subhalo_data['engulfs_Earth'] = (
            self.get_parameter('R_s') > self.get_parameter('D_Earth'))

        return

    def xx(self, mmax, mmin, root):
        return (self.number_subhalos(
            x_min=np.max((mmin, 1e-20)), x_max=mmax,
            force_no_fraction=True) - root)

    def store_dict_as_hdf(self, group, d):

        for key, value in d.items():
            if value is None:
                group.create_group(key)
                continue

            # Case 1: nested dict
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self.store_dict_as_hdf(subgroup, value)
                continue

            # Case 2: callable: convert to string
            if callable(value):
                value = inspect.getsource(value).strip()

            # Case 3: list of callables or mixed lists
            if isinstance(value, list):
                # create subgroup and save each element
                list_group = group.create_group(key)
                for i, element in enumerate(value):
                    item_name = f"item_{i}"

                    if callable(element):
                        element = inspect.getsource(element).strip()

                    if isinstance(element, dict):
                        item_group = list_group.create_group(item_name)
                        self.store_dict_as_hdf(item_group, element)
                    else:
                        if isinstance(element, str):
                            data = np.string_(element)
                            dtype = h5py.string_dtype(encoding="utf-8")
                        else:
                            data = element
                            dtype = None
                        list_group.create_dataset(item_name, data=data,
                                                  dtype=dtype)
                continue

            # Case 4: primitive value
            if isinstance(value, str):
                data = np.string_(value)
                dtype = h5py.string_dtype("utf-8")
            else:
                data = value
                dtype = None

            if key in group:
                del group[key]
            group.create_dataset(key, data=data, dtype=dtype)

    def interior_full_repop(self):
        self.logging_info('Enter loop of iterations.')

        with h5py.File(self.path_output + 'fullrepop_'
                       + self.configuration + '.h5', 'a') as f:
            input_group = f.create_group(f'inputs')
            aaa = copy.deepcopy(self.input_dict_strings)
            aaa['configurations'] = aaa[
                'configurations'][self.configuration]
            self.store_dict_as_hdf(input_group, aaa)

            for iter_idx in range(self._its):

                if iter_idx % self._prntfrq == 0:
                    print('    %s %s: it %d' % (
                        time.strftime(
                            ' %Y-%m-%d %H:%M:%S', time.gmtime()),
                        self.configuration, iter_idx))
                    progress = open(self.path_output + 'progress_'
                                    + self.configuration + '.txt', 'a')
                    progress.write(
                        self.configuration
                        + ', iteration ' + str(iter_idx))
                    progress.write(
                        '        %.3f  %s\n' %
                        (memory_usage_psutil(),
                         time.strftime(
                             ' %Y-%m-%d %H:%M:%S', time.gmtime())))
                    progress.close()

                iter_group = f.create_group(f'iteration_{iter_idx}')
                datasets = {}

                # We calculate our subhalo population in bins
                self._m_min = self.RangeMin

                while self._m_min < self.RangeMax:

                    self.calculate_characteristics_subhalo()
                    self.logging_info('All characterizations done.')

                    for key, array in self.subhalo_data.items():
                        if key not in iter_group.keys():
                            datasets[key] = iter_group.create_dataset(
                                key,
                                shape=(0,),
                                maxshape=(None,),
                                dtype=array.dtype,
                                chunks=True,
                                compression='gzip'
                            )
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''

                        dataset = datasets[key]
                        current_size = dataset.shape[0]
                        new_size = current_size + len(array)
                        dataset.resize((new_size,))
                        dataset[current_size:new_size] = array
                    self.logging_info('All characterizations saved,'
                                      'loop continues.')
                    self._m_min = self._m_max
            f.flush()
        return

    def interior_brightest(self):
        self.logging_info('Enter loop of iterations.')

        with h5py.File(self.path_output + 'brightest_'
                       + self.configuration + '.h5', 'a') as f:
            input_group = f.create_group(f'inputs')
            aaa = copy.deepcopy(self.input_dict_strings)
            aaa['configurations'] = aaa[
                'configurations'][self.configuration]
            self.store_dict_as_hdf(input_group, aaa)

            datasets = {}

            for iter_idx in range(self._its):

                if iter_idx % self._prntfrq == 0:
                    print('    %s %s: it %d' % (
                        time.strftime(
                            ' %Y-%m-%d %H:%M:%S', time.gmtime()),
                        self.configuration, iter_idx))
                    progress = open(self.path_output + 'progress_'
                                    + self.configuration + '.txt', 'a')
                    progress.write(
                        self.configuration
                        + ', iteration ' + str(iter_idx))
                    progress.write(
                        '        %.3f  %s\n' %
                        (memory_usage_psutil(),
                         time.strftime(
                             ' %Y-%m-%d %H:%M:%S', time.gmtime())))
                    progress.close()

                iter_group = f.create_group(f'iteration_{iter_idx}')

                highest_dict = {}
                for ii in self.input_dict[
                    'repopulations']['params_to_order_by']:
                    highest_dict[ii] = {}

                # We calculate our subhalo population in bins
                self._m_min = self.RangeMin

                while self._m_min < self.RangeMax:

                    self.calculate_characteristics_subhalo()
                    self.logging_info('All characterizations done.')

                    for ii in self.input_dict[
                        'repopulations']['params_to_order_by']:
                        data_bright = self.get_parameter(ii)

                        if not self.input_dict[
                        'repopulations']['allow_break_Roche']:
                            if ('survives_Roche'
                                    not in self.subhalo_data.keys()):
                                self.subhalo_data['survives_Roche'] = (
                                    self.get_parameter('R_t')
                                    > self.get_parameter('R_s')
                                )
                            data_bright *= self.get_parameter(
                                'survives_Roche')

                        if not self.input_dict[
                            'repopulations']['allow_engulf_Earth']:
                            self.get_parameter('engulfs_Earth')
                            data_bright *= ~self.get_parameter(
                                'engulfs_Earth')

                        if self._number_highest < self._num_subhalos:
                            temp = np.argpartition(
                                -data_bright, self._number_highest)
                            highest_indexes = temp[:self._number_highest]

                            for key, array in self.subhalo_data.items():
                                if key not in highest_dict[ii].keys():
                                    highest_dict[ii][key] = (
                                        array[highest_indexes].copy()
                                    )
                                else:
                                    highest_dict[ii][key] = np.append(
                                        highest_dict[ii][key],
                                        array[highest_indexes])
                        else:

                            for key, array in self.subhalo_data.items():
                                if key not in highest_dict[ii].keys():
                                    highest_dict[ii][key] = (
                                        array.copy()
                                    )
                                else:
                                    highest_dict[ii][key] = np.append(
                                        highest_dict[ii][key], array)

                    self.logging_info('All characterizations saved,'
                                      'loop continues.')
                    self._m_min = self._m_max

                for ii in self.input_dict[
                    'repopulations']['params_to_order_by']:

                    self.logging_info('Order by highest: %s' % ii)

                    highest_group = iter_group.create_group(
                        f'highest_' + str(ii))

                    if len(highest_dict[ii][ii]) > self._number_highest:
                        temp = np.argpartition(
                            -highest_dict[ii][ii], self._number_highest)
                        highest_indexes = temp[:self._number_highest]

                        for key, array in highest_dict[ii].items():
                            datasets[key] = (
                                highest_group.create_dataset(
                                    key,
                                    data=array[highest_indexes],
                                    chunks=True,
                                    compression='gzip'
                                ))
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''

                    else:
                        for key, array in highest_dict[ii].items():
                            datasets[key] = (
                                highest_group.create_dataset(
                                    key,
                                    data=array,
                                    chunks=True,
                                    compression='gzip'
                                ))
                            try:
                                datasets[key].attrs['units'] = str(
                                    self.subhalo_data[key].unit)
                            except AttributeError:
                                datasets[key].attrs['units'] = ''

                self.logging_info('Finish ordering by highest.')
                print('Memory in use: %.1f MB' % memory_usage_psutil())
            f.flush()
        return

    # ----------- General formulas -------------------------------------
    def Rmax(self, Vmax=None, Cv=None, Mass=None, Cmass=None,
             density_profile=None, unit=None):
        """
        Calculate Rmax of a subhalo.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            Rmax of the subhalo given by the inputs.
        """
        if self._param_repopulation == 'Vmax':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')

            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
            r_max = Vmax / cosmo_H_0 * np.sqrt(2. / Cv)

        elif self._param_repopulation == 'Mass':
            if Mass is None:
                Mass = self.get_parameter('Mass')
            if Cmass is None:
                Cmass = self.get_parameter('Cmass')
            if density_profile is None:
                density_profile = self.input_dict['configurations'][
                    self.configuration]['internal_density_profile']

            RmaxoverrS = self.RmaxoverrS(density_profile=density_profile)
            r_max = (self.R_s(
                Mass=Mass, Cmass=Cmass, density_profile=density_profile)
                     * RmaxoverrS)

        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['Rmax']['unit']
            except KeyError:
                unit = 'kpc'

        return r_max.to(u.Unit(unit))

    def R_s(self, Mass=None, Cmass=None, Vmax=None, Cv=None,
            density_profile=None, unit=None):
        """
        Calculate scale radius (R_s) of a subhalo following the NFW
        analytical expression for a subhalo density profile.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            R_s of the subhalo given by the inputs.
        """
        if self._param_repopulation == 'Vmax':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')
            if density_profile is None:
                density_profile = self.input_dict['host']['density_profile']

            RmaxoverrS = self.RmaxoverrS(density_profile=density_profile)
            r_s = (self.Rmax(
                Vmax=Vmax, Cv=Cv, density_profile=density_profile)
                   / RmaxoverrS)

        elif self._param_repopulation == 'Mass':
            if Mass is None:
                Mass = self.get_parameter('Mass')
            if Cmass is None:
                Cmass = self.get_parameter('Cmass')

            rho_crit = self.input_dict['cosmo_constants']['rho_crit']
            delta = self.input_dict['cosmo_constants']['delta_overdensity']

            Rvir = (3. * Mass / (4. * np.pi * delta * rho_crit)) ** (1/3.)
            r_s = Rvir / Cmass

        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['R_s']['unit']
            except KeyError:
                unit = 'kpc'

        return r_s.to(u.Unit(unit))

    def R_t(self, Mass=None, Vmax=None, Cv=None, D_GC=None,
            density_profile_sub=None, unit=None):
        """
        Calculation of tidal radius (R_t) of a subhalo, following the
        NFW analytical expression for a subhalo density profile.

        Definition of R_t: 1603.04057 King radius pg 14

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.
        :param D_GC: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).

        :return: float or array-like [kpc]
            Tidal radius of the subhalo given by the inputs.
        """
        if D_GC is None:
            D_GC = self.get_parameter('D_GC')

        if self._param_repopulation == 'Vmax':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')
            if density_profile_sub is None:
                density_profile_sub = self.input_dict['configurations'][
                    self.configuration]['internal_density_profile']

            self.logging_info('Rt: enter')

            method_Vmax_Mass = self.input_dict['configurations'][
                    self.configuration]['relation_Mass_Vmax']

            if method_Vmax_Mass == 'M200_from_VmaxRmax':

                Rmax = self.Rmax(
                    Vmax=Vmax, Cv=Cv,
                    density_profile=density_profile_sub)
                self.logging_info('Rt: Rmax')
                c200 = self.C200_from_Cv(
                    Cv=Cv, density_profile=density_profile_sub)
                self.logging_info('Rt: c200')
                Mass = self.Mass_from_Vmax(
                    radius_normalized=c200, Vmax=Vmax, Rmax=Rmax,
                    density_profile=density_profile_sub,
                    method=method_Vmax_Mass)
            else:
                Mass = self.Mass_from_Vmax(
                    Vmax=Vmax, method=method_Vmax_Mass)

            self.logging_info('Rt: mass')

        elif self._param_repopulation == 'Mass':
            if Mass is None:
                Mass = self.get_parameter('Mass').copy()

        density_profile_host = self.input_dict['host']['density_profile']
        host_rho_0 = self.input_dict['host']['rho_0']
        host_r_s = self.input_dict['host']['r_s']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['R_t']['unit']
            except KeyError:
                unit = u.kpc

        M_host = self.M_encapsulated(
            radius=D_GC, rho_0=host_rho_0, r_s=host_r_s,
            density_profile=density_profile_host)
        self.logging_info('Rt: mass host')

        return (D_GC * (Mass / (3 * M_host)) ** (1/3.)
                ).to(u.Unit(unit))

    def M_encapsulated(self, radius, rho_0, r_s,
                       density_profile, unit=None):
        """
        Mass encapsulated up to a certain radius.
        We assume a known density profile for the (sub)halo.

        :param R: float or array-like [kpc]
            Radius up to which we integrate the density profile.

        :return: float or array-like [Msun]
            Host mass encapsulated up to R.
        """
        # if radius is None:
        #     radius = self.get_parameter('D_GC')
        # if rho_0 is None:
        #     rho_0 = self.input_dict['host']['rho_0']
        # if r_s is None:
        #     r_s = self.input_dict['host']['r_s']
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['M_encapsulated']['unit']
            except KeyError:
                unit = 'Msun'

        return (4 * np.pi * rho_0 * r_s ** 3
                * self.ff(radius/r_s, density_profile=density_profile)
                ).to(u.Unit(unit))

    def C200_from_Cv(self, Cv=None, density_profile=None):
        """
        Formula to find c200 knowing Cv to input in the Newton
        root-finding method.

        :param c200: float or array-like
            c200 of subhalo (concentration definition)
        :param Cv: float or array-like
            Cv of subhalo (concentration definition)

        :return: float or array-like
            The output will be 0 when you find the c200 for a
            specific Cv.
        """
        if Cv is None:
            Cv = self.get_parameter('Cv')
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        delta = self.input_dict['cosmo_constants']['delta_overdensity']
        RmaxoverrS = self.RmaxoverrS(density_profile=density_profile)

        if isinstance(Cv, u.Quantity):
            Cv.value

        def int_interior(c200i, Cvi):
            return (delta
                    * c200i ** 3 / self.ff(c200i, density_profile)
                    * self.ff(RmaxoverrS, density_profile)
                    / RmaxoverrS ** 3
                    - Cvi)

        self.logging_info('C200_from_Cv: enter')

        try:
            c200 = newton(int_interior, x0=40.0, args=[Cv])

        except: # (ValueError, TypeError):
            c200_min = newton(int_interior, x0=40.0,
                              args=[np.min(Cv)])
            c200_max = newton(int_interior, x0=40.0,
                              args=[np.max(Cv)])
            C200_array = np.geomspace(
                0.95 * c200_min, 1.05 * c200_max, num=500)
            Cv_array = (delta * C200_array ** 3
                        / self.ff(C200_array, density_profile)
                        * self.ff(RmaxoverrS, density_profile)
                        / RmaxoverrS ** 3)
            spline = UnivariateSpline(
                np.log10(Cv_array), np.log10(C200_array),
                s=0, k=1, ext=2)

            c200 = 10**spline(np.log10(Cv))

        self.logging_info('C200_from_Cv: newton calculated')

        return c200 * u.dimensionless_unscaled

    def Mass_from_Vmax(self, radius_normalized=None,
                       Vmax=None, Rmax=None,
                       density_profile=None,
                       unit=None, method=None):
        """
        Mass from a subhalo assuming a NFW profile.
        Theoretical steps in Moline16.

        :param Vmax: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.
        :param Rmax: float or array-like [kpc]
            Radius at which Vmax happens (from the subhalo center).
        :param c200: float or array-like
            Concentration of the subhalo in terms of mass.
        :return: float or array-like [Msun]
            Mass from the subhalo assuming a NFW profile.
        """
        if method is None:
            method = self.input_dict['configurations'][
                    self.configuration]['relation_Mass_Vmax']
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['mass_from_Vmax']['unit']
            except KeyError:
                unit = 'Msun'

        if method == 'M200_from_VmaxRmax':
            if Rmax is None:
                Rmax = self.get_parameter('Rmax')
            if density_profile is None:
                density_profile = self.input_dict['configurations'][
                    self.configuration]['internal_density_profile']

            self.logging_info('Mass_from_Vmax: enter')

            cosmo_G = self.input_dict['cosmo_constants']['G']
            Rmax_over_rs = self.RmaxoverrS(density_profile)

            yy = (Vmax ** 2 * Rmax / cosmo_G
                  * self.ff(radius_normalized, density_profile)
                  / self.ff(Rmax_over_rs, density_profile))

            self.logging_info('Mass_from_Vmax: yy calculated')
        else:
            if 'params' in method.keys():
                params = method['params']
            else:
                params = None
            if isinstance(Vmax, u.Quantity):
                Vmax = (Vmax.copy()).to(u.Unit(
                    self.input_dict['repopulations']['params_to_save'][
                        'Vmax']['unit'])).value
            yy = self.calculate_formula(
                    Vmax, formula=method['formula'], params=params)

            if not isinstance(yy, u.Quantity):
                yy *= u.Unit(unit)

        return yy.to(u.Unit(unit))

    def theta_s(self, D_Earth=None,
                Vmax=None, Cv=None, density_profile=None,
                Mass=None, Cmass=None,
                unit='degree'):
        # Angular size of subhalos (up to R_s)
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')

        if self._param_repopulation == 'Vmax':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')

            if density_profile is None:
                density_profile = self.input_dict['configurations'][
                    self.configuration]['internal_density_profile']

            yy = self.R_s(
                Vmax=Vmax, Cv=Cv, density_profile=density_profile)

        elif self._param_repopulation == 'Mass':
            if Mass is None:
                Mass = self.get_parameter('Mass')
            if Cmass is None:
                Cmass = self.get_parameter('Cmass')

            yy = self.R_s(
                Mass=Mass, Cmass=Cmass, density_profile=density_profile)

        return np.arctan(yy / D_Earth).to(u.Unit(unit))

    # ----------- Cv ---------------------------------------------------

    def Cv_Mol2021_redshift0_scattered(
            self, Vmax=None,
            c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028,
            sigma_scatter=0., **kwargs):
        # Median subhalo concentration depending on its Vmax and
        # its redshift (here z=0).
        # Moline et al. 2110.02097

        # Create a scatter in the concentration parameter of the
        # repopulated population.
        # Scatter in logarithmic scale, following a Gaussian distribution.
        #
        # :param C: float or array-like
        #     Concentration of a subhalo (according to the concentration
        #     law).
        # :return: float or array-like
        #     Subhalos with scattered concentrations.
        #
        # V - max radial velocity of a bound particle in the subhalo [km/s]
        if Vmax is None:
            Vmax = self.get_parameter('Vmax')
        ci = [c0, c1, c2, c3]
        Vmax = (Vmax * u.s / u.km).to(1)

        yy = ci[0] * (1 + (
            sum([ci[i + 1] * np.log10(Vmax) ** (i + 1)
                for i in range(3)])))
        try:
            scatter = 10 ** self.rng.normal(
                loc=0, scale=sigma_scatter, size=len(Vmax))
        except TypeError:
            scatter = 10 ** self.rng.normal(loc=0, scale=sigma_scatter)

        return yy * scatter * u.dimensionless_unscaled

    def C_200(self, Mass=None, D_GC=None, ci=None, **kwargs):
        if Mass is None:
            Mass = self.get_parameter('Mass')
        if D_GC is None:
            D_GC = self.get_parameter('D_GC')
        if ci is None:
            try:
                ci = self.input_dict['configurations'][
                self.configuration][self._concentr]['params']['ci']
            except (KeyError, TypeError):
                ci = [19.9, -0.195, 0.089, 0.089, -0.54]

        cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        R_vir_host = self.input_dict['host']['R_vir']

        yy = (Mass / (1e8 * u.Msun)
              * cosmo_H_0 / (100 * u.km / (u.s * u.Mpc))).to(1)

        return (ci[0] * (1 + sum([(ci[i + 1] * np.log10(yy)) ** (i + 1)
                                  for i in range(3)]))
                * (1 + ci[4] * np.log10((D_GC / R_vir_host).to(1))))

    # ----------- J-FACTORS --------------------------------------------
    def J_general(
            self, D_Earth=None,
            density_profile=None,
            calculate_from=None,
            integrate_up_to=None, unit=None,
            Vmax=None, Cv=None,
            Mass=None, Cmass=None,
            rho_0=None, r_s=None
    ):
        """
        J-factor enclosing whole subhalo as a function of the
        subhalo Vmax.

        :param V: float or array-like  [km/s]
            Maximum circular velocity inside a subhalo.
        :param D_Earth: float or array-like [kpc]
            Distance between the subhalo and the Earth.
        :param C: float or array-like
            Subhalo concentration.
        :param unit: str
            Change the output units of the Jfactors.

        :return: float or array-like
            Jfactor of a whole subhalo.
        """
        if calculate_from is None:
            calculate_from = self._calculate_from
        if calculate_from == 'Vmax_Cv':
            self._param_repopulation = 'Vmax'
        if calculate_from == 'mass_Cmass':
            self._param_repopulation = 'Mass'
        if integrate_up_to is None:
            integrate_up_to = self.input_dict['repopulations'][
                'params_to_save'][self._current_param_to_save][
                'integrate_up_to']
        if D_Earth is None:
            D_Earth = self.get_parameter('D_Earth')
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if integrate_up_to == 'R_s':
            radius_normalized = 1.

        elif integrate_up_to == 'R_t':
            radius_normalized = self.get_parameter('R_t').copy()

        elif integrate_up_to == 'whole':
            radius_normalized = 1000.

        elif isinstance(integrate_up_to, float):
            radius_normalized = (
                    D_Earth * np.tan(integrate_up_to * np.pi / 180.))

        else:
            raise ValueError(
                f'Error at initializing the integration bounds of'
                f' the Jfactor, with value {integrate_up_to}. '
                'Allowed values are: R_s, R_t, whole,'
                ' and a float that represents the angular extension.')

        yy = 1. / D_Earth ** 2.

        if calculate_from == 'Vmax_Cv':
            if Vmax is None:
                Vmax = self.get_parameter('Vmax')
            if Cv is None:
                Cv = self.get_parameter('Cv')

            cosmo_G = self.input_dict['cosmo_constants']['G']
            cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']

            Rmax_over_rs = self.RmaxoverrS(
                density_profile=density_profile)

            if (isinstance(integrate_up_to, float)
                    or integrate_up_to == 'R_t'):
                radius_normalized *= (
                        Rmax_over_rs / self.Rmax(
                    Vmax=Vmax, Cv=Cv, density_profile=density_profile))

            yy *= (cosmo_H_0 / 4. / np.pi / cosmo_G ** 2
                   * np.sqrt(Cv / 2) * Vmax ** 3
                   * Rmax_over_rs ** 3.
                   / self.ff(Rmax_over_rs, density_profile) ** 2.)

        elif calculate_from == 'rho0_rS':
            if rho_0 is None:
                rho_0 = self.get_parameter('rho_0')
            if r_s is None:
                r_s = self.get_parameter('r_s')

            if (isinstance(integrate_up_to, float)
                    or integrate_up_to == 'R_t'):
                radius_normalized /= r_s

            yy *= 4. * np.pi * rho_0 ** 2. * r_s ** 3.

        elif calculate_from == 'mass_Cmass':
            if Mass is None:
                Mass = self.get_parameter('Mass')
            if Cmass is None:
                Cmass = self.get_parameter('Cmass')

            rho_crit = self.input_dict['cosmo_constants']['rho_crit']
            delta = self.input_dict['cosmo_constants']['delta_overdensity']

            if (isinstance(integrate_up_to, float)
                    or integrate_up_to == 'R_t'):
                Rvir = (3. * Mass / (4. * np.pi * delta * rho_crit)
                        ) ** (1/3.)
                radius_normalized /= (Rvir / Cmass)
            yy *= (delta / 3. * rho_crit * Mass * Cmass ** 3.
                   / self.ff(Cmass, density_profile) ** 2.)

        else:
            raise ValueError(
                f'Error at initializing the way to calculate the Jfactor.'
                + f' with value {calculate_from}.'
                  'Allowed values are: mass_Cmass, Vmax_Rmax, and rho0_rS')

        if isinstance(radius_normalized, u.Quantity):
            radius_normalized = radius_normalized.to(1).value
        yy *= self.fff(radius_normalized, density_profile)

        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save'][self._current_param_to_save][
                    'unit']
            except (KeyError, AttributeError):
                unit = 'GeV2 cm-5'

        return yy.to(u.Unit(unit), equivalencies=mass_energy2)

    # ------------------------------------------------------------------
    # ------- Internal density profile ---------------------------------
    def ff(self, x, density_profile=None):

        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, u.Quantity):
            x = x.to(1).value

        if density_profile == 'NFW':
            return np.log(1. + x) - x / (1. + x)

        elif density_profile == 'Burkert':
            return (0.25 * np.log(1. + x ** 2.)
                    + 0.5 * np.log(1 + x)
                    - 0.5 * np.arctan(x))

        else:
            if 'ff_spline' in density_profile.keys():
                try:
                    return density_profile['ff_spline'](x)
                except ValueError:
                    try:
                        formula = density_profile['formula']
                        try:
                            params = density_profile['params']
                        except KeyError:
                            params = []

                        int_total = np.zeros_like(x)

                        def integrand(x_prime):
                            rho_x = self.calculate_formula(
                                x_prime, formula, params)
                            return x_prime ** 2 * rho_x

                        if isinstance(x, float) or isinstance(x, int):
                            int_total = quad(
                                lambda x_prime: integrand(x_prime),
                                a=0., b=x)[0]

                        elif isinstance(x, list) or isinstance(x, np.ndarray):
                            for ni, xi in enumerate(x):
                                int_total[ni] = quad(
                                    lambda x_prime: integrand(x_prime),
                                    a=0., b=xi)[0]

                        return int_total

                    except Exception as e:
                        raise ValueError(
                            f'Error evaluating formula ' + formula
                            + f' with parameters {params}: {e}')
            else:
                self.logging_info('Creating spline for ff(x).')
                try:
                    formula = density_profile['formula']
                    try:
                        params = density_profile['params']
                    except KeyError:
                        params = []

                    xx_array = np.linspace(
                        0.,
                        np.max((150.,
                                self.input_dict['host']['R_vir']
                                / self.input_dict['host']['r_s'])),
                        num=2000
                    )
                    int_total = np.zeros_like(xx_array)

                    def integrand(x_prime):
                        rho_x = self.calculate_formula(
                            x_prime, formula, params)
                        return x_prime ** 2 * rho_x

                    for ni, xi in enumerate(xx_array):
                        int_total[ni] = quad(
                            lambda x_prime: integrand(x_prime),
                            a=0., b=xi)[0]

                    density_profile['ff_spline'] = UnivariateSpline(
                        xx_array, int_total, k=1, s=0
                    )

                    return density_profile['ff_spline'](x)

                except Exception as e:
                    raise ValueError(
                        f'Error evaluating formula ' + formula
                        + f' with parameters {params}: {e}')

    def fff(self, x, density_profile=None):
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, u.Quantity):
            x = x.to(1).value

        if density_profile == 'NFW':
            return (1 - 1 / (1 + x) ** 3.) / 3.

        elif density_profile == 'Burkert':
            return 0.25 * (2. - 1 / (1 + x)
                           - 1 / (1 + x ** 2) - np.arctan(x))

        else:
            if 'fff_spline' in density_profile.keys():
                try:
                    return density_profile['fff_spline'](x)
                except ValueError:
                    try:
                        formula = density_profile['formula']
                        try:
                            params = density_profile['params']
                        except KeyError:
                            params = []

                        int_total = np.zeros_like(x)

                        def integrand(x_prime):
                            rho_x = self.calculate_formula(
                                x_prime, formula, params)
                            return x_prime ** 2 * rho_x ** 2

                        if isinstance(x, float) or isinstance(x, int):
                            int_total = quad(
                                lambda x_prime: integrand(x_prime),
                                a=0., b=x)[0]

                        elif isinstance(x, list) or isinstance(x, np.ndarray):
                            for ni, xi in enumerate(x):
                                int_total[ni] = quad(
                                    lambda x_prime: integrand(x_prime),
                                    a=0., b=xi)[0]

                        return int_total

                    except Exception as e:
                        raise ValueError(
                            f'Error evaluating formula ' + formula
                            + f' with parameters {params}: {e}')
            else:
                self.logging_info('Creating spline for fff(x).')
                try:
                    formula = density_profile['formula']
                    try:
                        params = density_profile['params']
                    except KeyError:
                        params = []

                    xx_array = np.linspace(
                        0.,
                        np.max((150.,
                                self.input_dict['host']['R_vir']
                                / self.input_dict['host']['r_s'])),
                        num=2000
                    )
                    int_total = np.zeros_like(xx_array)

                    def integrand(x_prime):
                        rho_x = self.calculate_formula(
                            x_prime, formula, params)
                        return x_prime ** 2 * rho_x

                    for ni, xi in enumerate(xx_array):
                        int_total[ni] = quad(
                            lambda x_prime: integrand(x_prime),
                            a=0., b=xi)[0]

                    density_profile['fff_spline'] = UnivariateSpline(
                        xx_array, int_total, k=1, s=0
                    )

                    return density_profile['fff_spline'](x)

                except Exception as e:
                    raise ValueError(
                        f'Error evaluating formula ' + formula
                        + f' with parameters {params}: {e}')

    def RmaxoverrS(self, density_profile=None):

        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        try:
            return density_profile['RmaxoverrS']

        except (TypeError, KeyError):

            if density_profile == 'NFW':
                return 2.16257584237016

            elif density_profile == 'Burkert':
                return 3.244597456471571

            else:
                try:
                    def funcc(xx):
                        return -self.ff(xx, density_profile) / xx

                    argwhere = minimize(funcc, x0=3.)

                    density_profile['RmaxoverrS'] = argwhere['x'][0]
                    return argwhere['x'][0]

                except Exception as e:
                    raise ValueError(
                        f'Error evaluating density profile'
                        + density_profile + f': {e}')

    # ------------------------------------------------------------------
    # ----------- SRD --------------------------------------------------
    def srd_constant(self, xx, args):
        return args * np.ones_like(xx)

    def srd_exponential(self, xx, exp_fit, last_subhalo):

        try:
            xx = xx.to(u.kpc).value
        except AttributeError:
            xx = xx
        last_subhalo = last_subhalo.to(u.kpc).value

        return (exp_fit[1] * np.exp(exp_fit[0] / xx * exp_fit[2])
                * (xx >= last_subhalo))

    # ------------------------------------------------------------------
    # ----------- SHVF -------------------------------------------------
    def power_law(self, Vmax, V0, slope):
        """
        SubHalo Velocity Function (SHVF) - number of subhalos as a
        function of Vmax. Power law formula.
        Definition taken from Grand 2012.07846.

        :param Vmax_array: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.

        :return: float or array-like
            Number of subhalos defined by the Vmax input.
        """
        return 10 ** V0 * Vmax ** slope

    def number_subhalos(self, x_min, x_max, force_no_fraction=None):

        formula_SHiF = self.input_dict['configurations'][
                self.configuration]['SHVF']['formula']
        params_SHiF = self.input_dict['configurations'][
                self.configuration]['SHVF']['params']

        fraction = 1.
        if not force_no_fraction:
            if self.input_dict['repopulations']['use_spherical_shells']:
                if self._param_repopulation == 'Vmax':
                    method = self.input_dict['configurations'][
                            self.configuration]['relation_Mass_Vmax']
                    if method == 'M200_from_VmaxRmax':
                        formula_concentr = self.input_dict[
                            'configurations'][self.configuration][
                            'Cv']['formula']

                        params_concentr = self.input_dict[
                            'configurations'][self.configuration][
                            'Cv']['params']
                        if isinstance(params_concentr, dict):
                            params_concentr = params_concentr.copy()
                            for i in params_concentr.keys():
                                if 'scatter' in i:
                                        params_concentr[i] = 0
                        if isinstance(formula_concentr, str):
                            formula_concentr = formula_concentr.replace(
                                'Vmax','xx')

                        cv_mean = self.calculate_formula(
                                x_max * u.km / u.s,
                                formula_concentr, params_concentr)
                        c200 = self.C200_from_Cv(Cv=cv_mean)
                        Rmax = self.Rmax(
                            Vmax=x_max * u.km / u.s, Cv=cv_mean)

                        M = self.Mass_from_Vmax(
                            radius_normalized=c200,
                            Vmax=x_max * u.km / u.s, Rmax=Rmax,
                            method='M200_from_VmaxRmax')
                    else:
                        M = self.Mass_from_Vmax(
                            Vmax=x_max, method=method)

                elif self._param_repopulation == 'Mass':
                    M = x_max * u.Msun

                else:
                    raise ValueError(
                        f'Error evaluating which parameter the '
                        f'repopulation takes: '
                        + self._param_repopulation
                        + "\n    Values accepted for "
                          "input_dict['repopulations']["
                          "'param_to_repopulate']: Mass, Vmax")

                self._Rcut = self.R_Cut(M)
                fraction = self.dist_frac(self._Rcut)

        return int(np.rint(fraction * quad(
            self.calculate_formula,
            a=x_min, b=x_max, args=(formula_SHiF, params_SHiF))[0]))

    def dist_frac(self, Rcut):

        formula_SRD = self.input_dict['configurations'][
                self.configuration]['SRD']['formula']
        params_SRD = self.input_dict['configurations'][
                self.configuration]['SRD']['params']
        R_vir = self.input_dict['host']['R_vir']

        try:
            units = self.input_dict[
                    'repopulations']['params_to_save']['D_GC']['unit']
        except KeyError:
            units = u.kpc

        R_vir = R_vir.to(units).value
        Rcut = Rcut.to(units).value
        dist_Earth_GC = self._dist_Earth_GC.to(units).value

        if Rcut < dist_Earth_GC:
            return (quad(
                self.calculate_formula,
                a=dist_Earth_GC - Rcut, b=dist_Earth_GC + Rcut,
                args=(formula_SRD, params_SRD))[0]
                    / quad(
                self.calculate_formula, a=0., b=R_vir,
                args=(formula_SRD, params_SRD))[0])

        elif Rcut + dist_Earth_GC >= R_vir:
            return 1.

        else:
            return (quad(
                self.calculate_formula,
                a=0., b=dist_Earth_GC + Rcut,
                args=(formula_SRD, params_SRD))[0]
                    / quad(
                self.calculate_formula, a=0., b=R_vir,
                args=(formula_SRD, params_SRD))[0])

    def R_Cut(self, Mass, density_profile=None, unit=None):
        # max radial dist. from earth at which subhalo of mass M might be observed

        Dgc_D = self.input_dict['host']['Draco']['D_GC']
        Mass_D = self.input_dict['host']['Draco']['Mass']
        C_D_corr = self.input_dict['host']['Draco']['Cdelta']
        percentage = self.input_dict['host']['Draco']['percentage_accepted']
        distance_subhalo_cutoff = self.input_dict['host']['Draco'][
            'distance_subhalo_cutoff']
        if density_profile is None:
            density_profile = self.input_dict['configurations'][
                self.configuration]['internal_density_profile']

        if unit is None:
            try:
                unit = self.input_dict['repopulations'][
                    'params_to_save']['D_GC']['unit']
            except KeyError:
                unit = 'u.kpc'

        # cutoff is R = .1, i.e. 10%
        C = self.C_200(Mass, distance_subhalo_cutoff)
        C_D_corr1 = self.C_200(Mass_D, Dgc_D)
        print('C_D_corr, C', C_D_corr1, C_D_corr, C)
        return ((Dgc_D**2 * Mass * C**3
                 * self.ff(C_D_corr, density_profile=density_profile)**2
                 / (percentage * Mass_D * C_D_corr**3
                    * self.ff(C, density_profile=density_profile)**2)
                 ) ** .5).to(u.Unit(unit))
