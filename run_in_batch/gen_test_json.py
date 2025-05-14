import itertools
import json
import copy
import os
import numpy as np
current_path = os.path.abspath(__file__)
proj_abs_path = os.path.abspath(f"{current_path}/../../../..")
def generate_combinations(base_params, repeat_times=1):
    """
    Generate all possible parameter combinations from a dictionary of parameter lists and repeat each combination.
    
    :param base_params: A dictionary where the values are lists of possible values for each parameter.
    :param repeat_times: Number of times each experiment configuration should be repeated.
    :return: A list of dictionaries, each representing one combination of parameters.
    """
    # Extract parameter keys and value lists
    keys = base_params.keys()
    value_lists = base_params.values()
    
    # Generate all combinations of parameter values
    combinations = list(itertools.product(*value_lists))
    
    # Convert combinations into a list of dictionaries and repeat them
    experiments = []
    for combination in combinations:
        experiment_params = dict(zip(keys, combination))
        for _ in range(repeat_times):  # Repeat each combination 'repeat_times' times
            experiments.append(copy.deepcopy(experiment_params))
    
    return experiments

def generate_log_interval_list(start, end, num):
    """
    生成在指定区间内的对数间隔列表。
    
    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。
    
    返回:
    list: 对数间隔的列表，包含指定数量的值。
    """
    return np.logspace(np.log10(start), np.log10(end), num=num).tolist()

def generate_lin_interval_list(start, end, num):
    """
    生成在指定区间内的线性间隔列表。

    参数:
    start (int or float): 区间开始值。
    end (int or float): 区间结束值。
    num (int): 列表中值的数量。

    返回:
    list: 线性间隔的列表，包含指定数量的值。
    """
    return np.linspace(start, end, num=num).tolist()


test_lable = "Mist_SD21_Wikiart_random50_r4p8p12p16"
params_options = {
    "EXPERIMENT_NAME": ["Mist"],
    "data_path": [f"{proj_abs_path}/datasets"],
    "dataset_name":["wikiart-data"],
    "data_id":[i for i in range(50)],
    # "data_id":[0],
    # "data_id":[0,1],
    "r": [4,8,12,16],
    "attack_steps": [100],
    # "mixed_precision":['fp16'],
    # "model_path":['/data/home/yekai/github/MetaCloak-local/SD/v1-5-pruned.ckpt'],
    "model_path":['/data/home/yekai/github/MetaCloak-local/SD/v2-1_512-ema-pruned.ckpt'],
    # "model_config":['/data/home/yekai/github/DiffAdvPerturbationBench/Algorithms/mist/configs/stable-diffusion/v1-inference-attack.yaml'],
    "model_config":['/data/home/yekai/github/DiffAdvPerturbationBench/Algorithms/mist/configs/stable-diffusion/v2-inference-v-attack.yaml'],
    # "a painting"  'a photo'
    "concept_prompt":['a painting'],
    "mode": [2],
    "rate": [1],
    "block_num": [1],
    "input_size": [512],
}

# Number of times to repeat each configuration
repeat_times = 1


for key, value in params_options.items():
    use_log = True
    if type(value) is list:
        continue
    if value.startswith("lin:") or value.startswith("log:"):
        if value.startswith("lin:"):
            use_log = False
        args = value.split(":")[1:]
        assert len(args) == 3, "Invalid format for linear/log interval list"
        start, end, num = map(float, args)
        num = int(num)
        if use_log:
            params_options[key] = generate_log_interval_list(start, end, num)
        else:
            params_options[key] = generate_lin_interval_list(start, end, num)

# Generate all combinations
experiments = generate_combinations(params_options, repeat_times=repeat_times)

# Wrap in "untest_args_list" for proper JSON structure
output = {"test_lable":test_lable ,"settings":params_options,"untest_args_list": experiments}

# Print the generated combinations as JSON
print(json.dumps(output, indent=4))

py_path = os.path.dirname(os.path.abspath(__file__))
# Save to a JSON file
with open(f"{py_path}/{test_lable}.json", "w") as outfile:
    json.dump(output, outfile, indent=4)
