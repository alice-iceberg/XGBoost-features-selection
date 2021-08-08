import concurrent.futures

import ml
from tools import symptoms


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
        results = [executor.submit(ml.xgboost_algorithm, [14, symptom]) for symptom in symptoms]

    for f in concurrent.futures.as_completed(results):
        print(f.result())


if __name__ == '__main__':
    main()
