from cc_tmcn.model.model_uwb import Model_UWB
from cc_tmcn.model.model_5G import Model_5G

models = {

            "Model_UWB": lambda path: Model_UWB(path, "Model_UWB", epochs=20000),
            "Model_5G": lambda path: Model_5G(path, "Model_5G", epochs=20000),
            
        }
        
def load_model(model_name, model_path):
    """
    Loads model from the given data path

    Parameters
    ----------
    model_name : String
        Name of the model
        
    model_path : String
        Path to the model files

    Returns
    -------
    model : 
        DL model

    """
    
    if model_name in list(models.keys()):
        model = create_model(model_name, model_path)
        
        if model.load() == False:
            print("Load model " + model_name + " failed!")
            return None
        
        return model
    else:
        print("No trained model with name " + model_name + " available. Do training before prediction!")
        return None

def create_model(name, model_path):
    """
    Creates a model in the given data path

    Parameters
    ----------
    name : String
        Name of the model
        
    model_path : String
        Path to the model files

    Returns
    -------
    model : 
        DL model

    """
    if name in models:
        return models[name](model_path)
    else:
        return None
