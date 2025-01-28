import mlflow
from typing import Any

def convert_weather(x):
    if ('rain' in x) or ('drizzle' in x) or ('Thundery' in x):
        return 'Rain'
    if x == 'Fog':
        return 'Mist'
    if x == 'Partly cloudy':
        return 'Cloudy'
    return x

def one_hot_weather(x):
  if x == 'Rain':
    return 0
  if x == 'Mist':
    return 1
  if x == 'Cloudy':
    return 2
  if x == 'Sunny':
    return 3
  if x == 'Overcast':
    return 5
  return 6



def create_experiment(name: str, artifact_location: str, tags: dict[str, Any]):
    
    try:
        exp_id = mlflow.create_experiment(
            name=name,
            artifact_location=artifact_location,
            tags=tags
        )
    
    except Exception as e:
        print(f"Experiment {name} already exists")
        exp_id = mlflow.get_experiment_by_name(name).experiment_id

    return exp_id


def get_mlflow_experiment(experiment_id: str= None, experiment_name: str= None):
    if experiment_id:
        exp = mlflow.get_experiment(experiment_id)
    elif experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Please provide either experiment_id or experiment_name")
    
    return exp