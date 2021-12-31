import configuration
import data.construct
from copy import deepcopy
import data.constants as dconstants
import data.util as dutil

cfg = deepcopy(configuration.default_configuration)
cfg['data']['dataset'] = 'cora_full'
cfg['data']['num_dataset_splits'] = 2
cfg['data']['base_labels'] = ['Artificial_Intelligence/NLP', 'Artificial_Intelligence/Data_Mining','Artificial_Intelligence/Speech', 'Artificial_Intelligence/Knowledge_Representation','Artificial_Intelligence/Theorem_Proving', 'Artificial_Intelligence/Games_and_Search','Artificial_Intelligence/Vision_and_Pattern_Recognition', 'Artificial_Intelligence/Planning','Artificial_Intelligence/Agents','Artificial_Intelligence/Robotics', 'Artificial_Intelligence/Expert_Systems','Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning','Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance']
cfg['data']['corpus_labels'] = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning']
cfg['data']['train_labels'] = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning']
cfg['data']['setting'] = dconstants.TRANSDUCTIVE[0]
cfg['data']['ood_type'] = dconstants.LEFT_OUT_CLASSES[0]
cfg['data']['left_out_class_labels'] = 'all'
cfg['perturbation_budget'] = 0.1
cfg = configuration.get_experiment_configuration(cfg)

data_list, data_fixed, data_fixed_ood = data.construct.load_data_from_configuration(cfg['data'])

print('TEST FIXED\n', dutil.data_get_summary(data_fixed))
print('TEST OOD FIXED\n', dutil.data_get_summary(data_fixed_ood))
for i, datasets in enumerate(data_list):
    for name, dataset in datasets.items():
        print(f'SPLIT {i} DATASET {name}\n', dutil.data_get_summary(dataset))
