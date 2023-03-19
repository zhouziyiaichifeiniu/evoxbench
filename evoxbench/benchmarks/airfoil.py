import json
import numpy as np
from pathlib import Path
import os
from evoxbench.modules import SearchSpace, Evaluator, Benchmark, AirFoilMLPPredictor

__all__ = ['AirfoilSearchSpace', 'AirfoilEvaluator', 'AirfoilBenchmark']

abscissa = np.array([0.9999979395861412,
                     1.0000306786642674,
                     0.9769745474555531,
                     0.8780283504997537,
                     0.7280048476951797,
                     0.5471991659014869,
                     0.3600131357718379,
                     0.19173619198910435,
                     0.06509038613783182,
                     -0.002817709287095803,
                     -0.0028177092870960044,
                     0.0650903861378321,
                     0.1917361919891036,
                     0.3600131357718392,
                     0.5471991659014883,
                     0.7280048476951795,
                     0.8780283504997544,
                     0.9769745474555531,
                     1.0000306786642663,
                     0.9999979395861407])


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "airfoil" / name)


class AirfoilSearchSpace(SearchSpace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # initialize parameters

    @property
    def name(self):
        return 'AirfoilSearchSpace'

    def _sample(self):
        mean = 0
        std = 1e-3
        vertical = np.random.normal(mean, std, len(abscissa))
        samples = np.column_stack((abscissa, vertical))
        return samples

    def _encode(self, sampled_points):
        X = []
        for p in range(1, len(sampled_points) - 1):
            X.append(sampled_points[p][1])
        return np.array(X)

    def _decode(self, x):
        raise NotImplementedError

    def visualize(self, arch):
        raise NotImplementedError


class AirfoilEvaluator(Evaluator):
    def __init__(self,
                 objs='cl/cd',
                 model_path=get_path("mlp1.json"),
                 **kwargs):
        super().__init__(objs, **kwargs)
        self.objs = objs

        self.predictor = AirFoilMLPPredictor(pretrained=model_path)

    @property
    def name(self):
        return 'AirfoilEvaluator'

    def evaluate(self, archs,
                 true_eval=False,  # query the true (mean over three runs) performance
                 **kwargs):
        ans = self.predictor.predict(archs)
        return ans


class AirfoilBenchmark(Benchmark):
    def __init__(self, objs='cl/cd', **kwargs):
        search_space = AirfoilSearchSpace()
        evaluator = AirfoilEvaluator()
        super().__init__(search_space, evaluator, **kwargs)

    @property
    def name(self):
        return 'AirfoilBenchmark'

    def test(self):
        sample = self.search_space.sample(10)
        X = self.search_space.encode(sample)
        F = self.evaluator.evaluate(X)

        print()
        print(sample)
        print(X)
        print(F)

if __name__ == '__main__':
    benchmark = AirfoilBenchmark()
    benchmark.test()
