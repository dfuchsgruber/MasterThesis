import configuration
import data.build
from copy import deepcopy
import data.constants as dconstants
import data.util as dutil

cfg = deepcopy(configuration.default_configuration)
cfg['data']['dataset'] = 'cora_full'
cfg['data']['type'] = 'npz'
cfg['data']['num_dataset_splits'] = 2
cfg['data']['base_labels'] = ['Artificial_Intelligence/NLP', 'Artificial_Intelligence/Data_Mining','Artificial_Intelligence/Speech', 'Artificial_Intelligence/Knowledge_Representation','Artificial_Intelligence/Theorem_Proving', 'Artificial_Intelligence/Games_and_Search','Artificial_Intelligence/Vision_and_Pattern_Recognition', 'Artificial_Intelligence/Planning','Artificial_Intelligence/Agents','Artificial_Intelligence/Robotics', 'Artificial_Intelligence/Expert_Systems','Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning','Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance']
cfg['data']['corpus_labels'] = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning']
cfg['data']['train_labels'] = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning']
cfg['data']['setting'] = dconstants.HYBRID[0]
cfg['data']['ood_type'] = dconstants.LEFT_OUT_CLASSES[0]
#cfg['data']['ood_type'] = dconstants.PERTURBATION[0]
cfg['data']['ood_sampling_strategy'] = dconstants.SAMPLE_ALL[0]
cfg['data']['left_out_class_labels'] = ['Artificial_Intelligence/NLP', 'Artificial_Intelligence/Data_Mining', 'Artificial_Intelligence/Speech', 'Artificial_Intelligence/Knowledge_Representation', 'Artificial_Intelligence/Theorem_Proving', 'Artificial_Intelligence/Games_and_Search', 'Artificial_Intelligence/Vision_and_Pattern_Recognition', 'Artificial_Intelligence/Planning', 'Artificial_Intelligence/Agents', 'Artificial_Intelligence/Robotics', 'Artificial_Intelligence/Expert_Systems', 'Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance']
cfg['data']['drop_train_vertices_portion'] = 0.2
cfg['perturbation_budget'] = 0.1

if cfg['data']['ood_type'] in dconstants.PERTURBATION:
    cfg['data']['left_out_class_labels'] = []
    cfg['data']['base_labels'] = cfg['data']['train_labels']

cfg = configuration.get_experiment_configuration(cfg)

data_list, fixed_vertices = data.build.load_data_from_configuration(cfg['data'])

data_val = data_list[0][dconstants.VAL]
data_ood = data_list[0][dconstants.OOD_VAL]

dropped = set(data_ood[0].vertex_to_idx.keys()) - set(data_val[0].vertex_to_idx.keys())
print(f'Number dropped {len(dropped)} / {len(set(data_ood[0].vertex_to_idx.keys()))}.')
print(f'Fixed vertices {len(fixed_vertices)}')
print(f'Introduced vertices: {data_ood[0].is_train_graph_vertex.size(0) - data_ood[0].is_train_graph_vertex.sum()} / {data_ood[0].is_train_graph_vertex.size(0)}')
print(f'OOD vertices: {data_ood[0].is_out_of_distribution.sum()} / {data_ood[0].is_out_of_distribution.size(0)}')

for i, datasets in enumerate(data_list):
    # Check if actually all of the test vertices are used
    vertices_test = dutil.vertices_from_mask(datasets[dconstants.TEST][0].mask, datasets[dconstants.TEST][0].vertex_to_idx)
    vertices_test_ood = dutil.vertices_from_mask(datasets[dconstants.OOD_TEST][0].mask, datasets[dconstants.OOD_TEST][0].vertex_to_idx)
    assert (vertices_test | vertices_test_ood) == fixed_vertices

    for name, dataset in datasets.items():
        print(f'SPLIT {i} DATASET {name}\n', dutil.data_get_summary(dataset))
