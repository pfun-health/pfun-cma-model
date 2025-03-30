import sys
import json
from pathlib import Path
# append to path
pth = str(Path(__file__).parents[1])
if pth not in sys.path:
    sys.path.append(pth)
    print('appended %s to sys.path' % pth)
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.cma_model_params import CMAModelParams

cma = CMASleepWakeModel()
cmap = CMAModelParams()


def main():
    json_schema = json.dumps(cmap.model_json_schema(), indent=4)
    json_schema = json_schema.replace('\\n', '\n')
    print('\n' + json_schema, end='\n')


if __name__ == '__main__':
    main()
