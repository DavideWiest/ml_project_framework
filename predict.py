from ml_modules import ModelManager
import torch
import json
from pathlib import Path
from logger import logging
from model import BasicModel


class PredictorInterface():
    def __init__(self, model_class, data_mode="normal", model_feature_param="model", model_subdir=None):
        self.load_data_normal(data_mode)

        self.initialize_model(model_class, model_feature_param, model_subdir)

    def _get_feature_list_config(self, filename="config.json", key="model"):
        BASE_DIR = Path(__file__).resolve().parent
        CONFIG_JSON_PATH = BASE_DIR / filename

        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            mconfig = config[key]

        return [mconfig["input_features"], mconfig["hidden_units"], mconfig["output_features"]]

    def load_data_normal(self, data_mode):
        if data_mode == "normal":
            with open("data/example.csv", "r", encoding="utf-8") as f:
                examplefile = f.read().split("\n")

            self.exampledict = {line.split(",")[2]: line.split(",")[1] for line in examplefile[1:]}
        
        
    def initialize_model(self, model_class, model_feature_param="model", model_subdir=None):
        if isinstance(model_feature_param, str):
        
            features_out = self._get_feature_list_config("config.json", model_feature_param)
            
            input_features, hidden_units, output_features = features_out
        elif isinstance(model_feature_param, list) or isinstance(model_feature_param, tuple):
            assert len(model_feature_param) == 3, "Function parameter model_feature_param must be None or numerical list of [input_features, hidden_units, output_features]"
            input_features, hidden_units, output_features = model_feature_param

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mm = ModelManager(logging)

        self.model = model_class(input_features, output_features, hidden_units).to(self.device)
        self.statedict, path = self.mm.load(name=self.model.__class__.__name__, load_best_metric="loss", subdir=model_subdir)
        
        if self.statedict != None:
            self.model.load_state_dict(self.statedict)
        else:
            logging.info("Warning: No saved state dicts found")

    def predict(self, filepath="predict_params.json", do_print=True):
        with open(filepath, "r", encoding="utf-8") as f:
            file = json.load(f)

        params = []

        # make customized execution here
        # get parameters, pass into model as nums

        y_pred = self.pass_into_model(params)
        
        if do_print:
            print(y_pred)

        return y_pred
        

    def pass_into_model(self, params):
        params = [float(i) for i in params]
        params = torch.Tensor(params).to(self.device)

        y_pred = self.model(params)
        return y_pred


if __name__ == "__main__":
    pi = PredictorInterface()
    pi.predict()