from mmengine.registry import Registry

ENVIRONMENTS = Registry('environments', scope='fluxthemis')
EVALUATORS = Registry('evaluators', scope='fluxthemis')
MODEL_CLIENTS = Registry('model_clients', scope='fluxthemis')
RUNNERS = Registry('runners', scope='fluxthemis')
