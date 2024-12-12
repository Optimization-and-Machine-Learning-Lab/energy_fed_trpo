import os
import wandb
from dotenv import load_dotenv

class GeneralLogger:
    
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeneralLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):

        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self.wdb_log = False
        self.csv_log = False
        self.console_log = False
        self.logging_path = './logs/'
        self.exp_config = None

    def setup(self, config):

        # Load dotenv file
        current_path = os.path.abspath(os.path.dirname(__file__))
        load_dotenv(dotenv_path=os.path.join(current_path, "../../.env"))

        self._initialized = True
        self.wdb_log = config.get("wdb_log", False)
        self.csv_log = config.get("csv_log", False)
        self.console_log = config.get("console_log", False)
        self.logging_path = config.get("logging_path", './logs/')
        self.exp_config = config.get("exp_config", None)

        # Wandb related configurations

        if self.wdb_log:

            self.wdb_project_name = str(os.environ.get("WANDB_PROJECT_NAME"))
            self.wdb_entity_name = str(os.environ.get("WANDB_ENTITY"))
            self.wdb_tags = config.get("tags", [])
            self.wdb_run = config.get("wdb_run", wandb.run)
            self.wdb_run_id = config.get("wdb_run_id", wandb.run.id if wandb.run else None)

            wandb.login(key=str(os.environ.get("WANDB_KEY")))
            
            self.wdb_run = wandb.init(
                project=self.wdb_project_name,
                entity=self.wdb_entity_name,
                config=self.exp_config,
                tags=self.wdb_tags,
                resume='allow' if self.wdb_run_id else None,
                id=self.wdb_run_id,
            )
 
    def dict_to_csv(self, data: dict):

        file_path = f'{self.logging_path}log.csv' if not self.wdb_log else f'{self.wdb_run.dir}/log.csv'
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, "w") as csv_file:

            # Write headers if file does not exist
            if not file_exists:
                csv_file.write(",".join(data.keys()) + "\n")
            
            # Write values
            csv_file.write(",".join(str(data[key]) for key in data.keys()) + "\n")

    def log(self, data: dict, step: int = None, commit: bool = False):

        if self.wdb_log:
            wandb.log(data, step=step, commit=commit)

        if self.console_log:
            for key, value in data.items():
                print(f"{key}: {value} \n", end="", flush=True)

        if self.csv_log:
            self.dict_to_csv(data)

    def log_line_series(self, name, xs, ys, keys, title, x_name):
        if self.wdb_log:
            wandb.log({
                f'{name}' : wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title=title,
                    xname=x_name,
                )
            })

    def watch_model(self, models):
        if self.wdb_log:
            wandb.watch(models=models)

    def wdb_save_model(self):
        if self.wdb_log:
            wandb.save('model.pt')

    def wdb_load_model(self):
        if self.wdb_log:
            return wandb.restore('model.pt', run_path=f'{self.wdb_entity_name}/{self.wdb_project_name}/{self.wdb_run.id}')

    def wdb_write_summary(self, key, value):
        if self.wdb_log:
            self.wdb_run.summary[key] = value

    def finish(self):
    
        if self.wdb_log:
            wandb.finish()

        # Write the experiment configuration to a file

        file_path = f'{self.logging_path}exp_config.txt' if not self.wdb_log else f'{self.wdb_run.dir}exp_config.txt'

        with open(file_path, 'w') as f:
            f.write(str(self.exp_config))