import configparser
from azureml.core import Workspace

config = configparser.ConfigParser()
config.read('config.ini')

ws = Workspace.create(
    name=config['azure']['workspace_name'],
    subscription_id=config['azure']['subscription_id'],
    resource_group=config['azure']['resource_group'],
    create_resource_group=True,
    location=config['azure']['location'],
)
