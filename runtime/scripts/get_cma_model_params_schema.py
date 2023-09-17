import json
from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from chalicelib.engine.cma_model_params import CMAModelParams

cma = CMASleepWakeModel()
cmap = CMAModelParams()


def main():
    json_schema = json.dumps(cmap.model_json_schema(), indent=4)
    json_schema = json_schema.replace('\\n', '\n')
    print('\n' + json_schema, end='\n')


if __name__ == '__main__':
    main()
