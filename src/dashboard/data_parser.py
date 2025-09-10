import glob
import os
import pandas as pd
import numpy as np

from src.calibration import calculate_calibration, calculate_miscalibration_area


EXPERIMENTS = {
    'RANDOM': -2,
    'RANDOM_ENSEMBLE': -1,
    'pt1': 0.1,
}

def get_loss_landscapes(DATA_DIR: str):

    """
    Get the data
    """
    data_path = os.path.join(DATA_DIR, '**/*loss_landscape.csv')

    # Get the data
    data = []
    for file in glob.glob(data_path):
        experiment_type = EXPERIMENTS[os.path.dirname(file).split('/')[0].split('_')[-1]]
        model_id = file.split('/')[1].split('_')[-1]
        file_name = os.path.basename(file).split('.')[0].split('_')
        if len(file_name) == 6:
            date, target, omitted_element, data_partition = file_name[0], file_name[1], file_name[2], file_name[3]
        elif len(file_name) == 7:
            date, target, omitted_element, data_partition = file_name[0], file_name[1]+'_'+file_name[2], file_name[3], file_name[4]
        else:
            raise ValueError(f"File name {file_name} does not match expected format.")
        
        loss_landscape = np.array(pd.read_csv(file).values[:, 1:], dtype=float)

        try:
            raw_loss_landscape_coords = pd.read_csv(
                os.path.join(os.path.dirname(file), f'{date}_{target}_{omitted_element}_{data_partition}_model_loci.csv'),
                index_col=['Unnamed: 0']
                )
            raw_loss_landscape_coords = list(raw_loss_landscape_coords.values)
            loss_landscape_coords = []
            if len(raw_loss_landscape_coords) > 0:
                for i in range(len(raw_loss_landscape_coords)):
                    info = raw_loss_landscape_coords[i][0].strip('()').split(',')
                    loss_landscape_coords.append([int(info[0]), int(info[1]), float(info[2][-6:])])
            loss_landscape_coords = np.array(loss_landscape_coords, dtype=float)
        except FileNotFoundError:
            loss_landscape_coords = None
        

        data_dict = {
            'model_id': int(model_id),
            'date': date,
            'target': target,
            'omitted_element': omitted_element,
            'data_partition': data_partition,
            'loss_landscape': [loss_landscape],
            'model_loci': [loss_landscape_coords],
            'experiment_type': experiment_type,
        }

        data.append(pd.DataFrame(data_dict))
    
    data = pd.concat(data, ignore_index=True)

    return data




def get_predictions(DATA_DIR: str):
    data_path = os.path.join(DATA_DIR, '**/*llensemble.csv')

    # Get the data
    data = []
    for file in glob.glob(data_path):
        experiment_type = EXPERIMENTS[os.path.dirname(file).split('/')[0].split('_')[-1]]
        uncertainty = pd.read_csv(file).values
        model_id = file.split('/')[1].split('_')[-1]
        file_name = os.path.basename(file).split('.')[0].split('_')
        if len(file_name) == 6:
            target, omitted_element, data_partition = file_name[0], file_name[1], file_name[2]
        elif len(file_name) == 7:
            target, omitted_element, data_partition = file_name[0]+'_'+file_name[1], file_name[2], file_name[3]
        else:
            raise ValueError(f"File name {file_name} does not match expected format.")
        


        data_dict = {
            'model_id': int(model_id),
            'target': target,
            'omitted_element': omitted_element,
            'data_partition': data_partition,
            'ground_truth': uncertainty[0, :],
            'prediction':uncertainty[1, :],
            'standard_deviation': uncertainty[2, :],
            'experiment_type': experiment_type,
        }

        data.append(pd.DataFrame(data_dict))
    data = pd.concat(data, ignore_index=True)
    return data


def get_calibrations(DATA_DIR: str):
    data_path = os.path.join(DATA_DIR, '**/*calibration.csv')

    data = []

    for file in glob.glob(data_path):
        model_id = file.split('/')[1].split('_')[-1]
        dir_name = os.path.dirname(file)
        
        file_name = os.path.basename(file).split('.')[0].split('_')

        experiment_type = EXPERIMENTS[os.path.dirname(file).split('/')[0].split('_')[-1]]
        data_dict = pd.read_csv(file)
        data_dict['model_id'] = int(model_id)

        raw_obsv_pi = eval(data_dict['observed_pi'][0])
        raw_pred_pi = eval(data_dict['predicted_pi'][0])
        
        data_dict['observed_pi'] = [raw_obsv_pi]
        data_dict['predicted_pi'] = [raw_pred_pi]
        data_dict['experiment_type'] = experiment_type
        data.append(pd.DataFrame(data_dict))

    data = pd.concat(data, ignore_index=True)
    return data


def get_raw_predictions(DATA_DIR):
    data_path = os.path.join(DATA_DIR, '**/*raw.csv')

    data = []

    for file in glob.glob(data_path):
        model_id = file.split('/')[1].split('_')[-1]
        experiment_type = EXPERIMENTS[os.path.dirname(file).split('/')[0].split('_')[-1]]
        file_name = os.path.basename(file).split('.')[0].split('_')
        dir_name = os.path.dirname(file)
        # print(file_name)
        if len(file_name) == 5:
            target, omitted_element, data_partition = file_name[0], file_name[1], file_name[2]
        elif len(file_name) == 6:
            target, omitted_element, data_partition = file_name[0]+'_'+file_name[1], file_name[2], file_name[3]
        else:
            raise ValueError(f"File name {file_name} does not match expected format.")

        ground_truth_file = os.path.join(dir_name, f'{target}_{omitted_element}_{data_partition}_gt_std_llensemble.csv')
        # print(ground_truth_file)
        ground_truth = np.expand_dims(np.array(pd.read_csv(ground_truth_file).values[1, :]), axis=0)
        n_ground_truth = ground_truth.shape[1]
        # print('n gt ', n_ground_truth)
        data_dict = pd.read_csv(file, index_col=False)
        data_dict['model_id'] = int(model_id)
        data_dict['experiment_type'] = experiment_type
        data_dict['model_loss'] = np.mean(np.abs(data_dict.values[:, :n_ground_truth] - ground_truth), axis=1)

        data.append(pd.DataFrame(data_dict))

    data = pd.concat(data, ignore_index=True)
    return data