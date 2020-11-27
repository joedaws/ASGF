"""
    file: scribe.py
"""
import os
import glob
import torch
def hs_to_str(hs):
    """
    converts a list of hidden layer widths to a string

    [12,15] --> str(12.15)

    Inputs:
        hs -- list of hidden layer widths

    Outputs:
        name -- string representation of hs
    """
    return '.'.join(str(x) for x in hs)


class Scribe:
    """
    class for writting data to a file
    create file and necessary directories when they are not present
    """

    def write(self,path,mode='app',**kwargs):
        """
        write to the file the contents of kwargs at path or create then write if it
        doesn't already exist

        Inputs:
            path -- path to file to write in
            mode -- append or new mode
            kwargs -- data to write
        """
        # create head line and data line
        head_line = ", ".join(map(str,kwargs.keys()))
        data_line = ", ".join(map(str,kwargs.values()))

        if mode == 'app':
            try:
                # write to csv
                with open(path,'a') as f:
                    f.write(data_line+'\n')

            except:
                print('Scribe could not append')

        elif mode == 'new':
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # create file and write to csv
            with open(path,'w') as f:
                f.write(head_line+'\n')
                f.write(data_line+'\n')

class RLScribe(Scribe):
    """
    class for logging and saving Reinforcement Learning experiments
    using the dgs code

    Attributes:
        _save_root -- root directory for all data
        _env_name  -- type of reinforcement learning environment
        _arch_type -- type of network architecture to be used in run of experiments
    """

    def __init__(self, save_root, env_name, arch_type, alg_name='exp'):
        # root directory
        self._save_root = save_root

        # openAI gym environment type
        self._env_name = env_name

        # network architecture type
        self._arch_type = arch_type

        # name of the algorithm
        self._alg_name = alg_name

        # experiment counter
        self._exp_num = 0

    @property
    def exp_num(self):
        return self._exp_num

    @exp_num.setter
    def exp_num(self,new_num):
        self._exp_num = new_num

    def record_iteration_data(self,**kwargs):
        """
        path_to_csv = <save_root>/<environment_name>/<network_architecture>/<results>/
        """
        exp_num = self._exp_num
        # file path
        path_to_csv = f"{self._save_root}/{self._env_name}/{self._arch_type}/results/{self._alg_name}.{exp_num}.csv"

        if kwargs['iteration'] == 1:
            self.write(path_to_csv,mode='new',**kwargs)
        else:
            self.write(path_to_csv,**kwargs)

    def record_metadata(self,**kwargs):
        """
        path_to_metadata = <save_root>/<environment_name>/<network_architecture>/<results>/
        """
        exp_num = self._exp_num
        path_to_meta = f"{self._save_root}/{self._env_name}/{self._arch_type}/results/metadata.{self._alg_name}.{exp_num}.csv"

        self.write(path_to_meta,mode='new',**kwargs)

    def checkpoint(self,net,opt,it,best=False):
        """
        saves network weights

        Input:
            net  -- Pytorch neural network
            opt  -- Optimization clas
            it   -- iteration number
            best -- flag if this network achieved best performance so far

        Notes:
        path_to_weights = <save_root>/<environemnt_type>/<network_architecture>/<weights>

        network checkpoint file = str(exp<exp_num>.<it_num>.pkl)

        network best so far file = str(best.exp<exp_num>.<it_num>.pkl)
        """
        exp_num = self._exp_num
        path_to_weights = f"{self._save_root}/{self._env_name}/{self._arch_type}/weights/"

        # make the directories if necessary
        os.makedirs(os.path.dirname(path_to_weights), exist_ok=True)

        if best:
            # name of new best weights
            weights_name = f"best.exp.{exp_num}.it.{it}.pkl"

            # remove old best weights if they exist
            old_best = glob.glob(path_to_weights+f'best.exp.{exp_num}.*')
            if old_best:
                os.remove(old_best[0])

        else:
            weights_name = f"exp.{exp_num}.it.{it}.pkl"

        torch.save({
                   'it': it,
                   'best': best,
                   'exp_num': exp_num,
                   'net':net,
                   'opt':opt,
                   },
                   path_to_weights+weights_name)

class FScribe(Scribe):
    """
    class for logging and saving Functional optimization experiments
    using the dgs code

    Attributes:
        _save_root -- root directory for all data
        _fun_name  -- function name
    """

    def __init__(self,save_root,fun_name):
        # root directory
        self._save_root = save_root

        # openAI gym environment type
        self._fun_name = fun_name

        # experiment counter
        self._exp_num = 0

    @property
    def exp_num(self):
        return self._exp_num

    @exp_num.setter
    def exp_num(self,new_num):
        self._exp_num = new_num

    def record_iteration_data(self,**kwargs):
        """
        path_to_csv = <save_root>/<fun_name>/<results>/
        """
        exp_num = self._exp_num
        # file path
        path_to_csv = f"{self._save_root}/{self._fun_name}/results/exp.{exp_num}.csv"

        if kwargs['iteration'] == 1:
            self.write(path_to_csv,mode='new',**kwargs)
        else:
            self.write(path_to_csv,**kwargs)

    def record_metadata(self,**kwargs):
        """
        path_to_metadata = <save_root>/<fun_name>/<results>/
        """
        exp_num = self._exp_num
        path_to_meta = f"{self._save_root}/{self._fun_name}/results/metadata.exp.{exp_num}.csv"

        self.write(path_to_meta,mode='new',**kwargs)

    def checkpoint(self,x,opt,it,best=False):
        """
        saves network weights

        Input:
            x    -- minimizer point to be saved
            opt  -- Optimization clas
            it   -- iteration number
            best -- flag if this network achieved best performance so far

        Notes:
        path_to_weights = <save_root>/<environemnt_type>/<network_architecture>/<weights>

        network checkpoint file = str(exp<exp_num>.<it_num>.pkl)

        network best so far file = str(best.exp<exp_num>.<it_num>.pkl)
        """
        exp_num = self._exp_num
        path_to_minimizer = f"{self._save_root}/{self._fun_name}/minimizer/"

        # make the directories if necessary
        os.makedirs(os.path.dirname(path_to_minimizer), exist_ok=True)

        if best:
            # name of new best weights
            min_name = f"best.exp.{exp_num}.it.{it}.pkl"

            # remove old best weights if they exist
            old_best = glob.glob(path_to_minimizer+f'best.exp.{exp_num}.*')
            if old_best:
                os.remove(old_best[0])

        else:
            min_name = f"exp.{exp_num}.it.{it}.pkl"

        torch.save({
                   'min_x': x,
                   'best': best,
                   'exp_num': exp_num,
                   'opt':opt,
                   },
                   path_to_minimizer+min_name)



