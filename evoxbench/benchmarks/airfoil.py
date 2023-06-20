import json
import numpy as np
from pathlib import Path
import os
import pandas as pd
from evoxbench.modules import SearchSpace, Evaluator, Benchmark, AirFoilMLPPredictor
import matplotlib.pyplot as plt

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
    def __init__(self,
                 csv_path=get_path("20.csv"),
                 mean_path=get_path("mean_ctr.json"),
                 cl_cyclegan_path=get_path("cl_cyclegan.json"),
                 cl_cd_cyclegan_path=get_path("cl_cd_cyclegan.json"),
                 **kwargs):
        super().__init__(**kwargs)
        self.csv_path = csv_path
        checkpoints = json.load(open(mean_path, 'r'))
        weights = checkpoints['state_dicts']

        if not isinstance(weights, list):
            weights = [weights]

        model = {}
        for key, values in weights[0].items():
            model[key] = np.array(values)
        self.model = model
        self.cl_cyclegan = AirFoilMLPPredictor(pretrained=cl_cyclegan_path)
        self.cl_cd_cyclegan = AirFoilMLPPredictor(pretrained=cl_cd_cyclegan_path)

    # initialize parameters

    @property
    def name(self):
        return 'AirfoilSearchSpace'

    def _sample(self):
        x = np.random.randint(1, 6)
        mean = self.model['r0{}'.format(x)][1:-1]
        cov = self.model['cov{}'.format(x)]
        samples = np.random.multivariate_normal(mean, cov, 1)
        samples = np.insert(samples, 0, mean[0])
        samples = np.insert(samples, -1, mean[-1])
        samples = np.concatenate((abscissa.reshape(-1, 1), samples.reshape(-1, 1)), axis=1)
        return samples

    def _encode(self, sampled_points):
        X = [x[1] for x in sampled_points]
        X = X[1:-1]
        return np.array(X)

    def _decode(self, x):
        raise NotImplementedError

    def optimize(self, archs, mode):
        nurbs = self.encode(archs)

        if mode == 'cl':
            nurbs = self.cl_cyclegan.predict(nurbs)
        elif mode == 'cl/cd':

            nurbs = self.cl_cd_cyclegan.predict(nurbs)

        archs = archs[0]

        for i in range(1, len(archs)-1):
            archs[i][1] = nurbs[0][i-1]
        return [archs]

    def visualize(self, arch):
        data = np.array(pd.read_csv(self.csv_path, header=None))
        points = []
        X = []
        Y = []
        for i in range(len(data[0])):
            x = 0
            y = 0
            for j in range(len(data)):
                x += data[j][i] * arch[0][j][0]
                y += data[j][i] * arch[0][j][1]
            X.append(x)
            Y.append(y)
            points.append([x, y])
        plt.figure(figsize=(13.22, 13.30))
        plt.ylim([-0.1, 0.1])
        plt.xlim([-0.1, 1.1])
        plt.plot(X, Y, 'k')
        plt.axis('off')
        plt.show()


class AirfoilEvaluator(Evaluator):
    def __init__(self,
                 objs='cl&cl/cd',
                 cl_cd_model_path=get_path("cl_cd_mlp.json"),
                 cl_model_path=get_path("cl_mlp.json"),

                 **kwargs):
        super().__init__(objs, **kwargs)
        self.objs = objs
        self.cl_cd_predictor = AirFoilMLPPredictor(pretrained=cl_cd_model_path)
        self.cl_predictor = AirFoilMLPPredictor(pretrained=cl_model_path)

    @property
    def name(self):
        return 'AirfoilEvaluator'

    def evaluate(self, archs,
                 true_eval=False,  # query the true (mean over three runs) performance
                 **kwargs):
        cl = self.cl_predictor.predict(archs)
        cl_cd = self.cl_cd_predictor.predict(archs)
        return [list(cl[0]), cl_cd[0][0]]

    def cl_visualize(self, archs):
        x = np.arange(-2, 4.002, 0.01)
        plt.plot(x, archs)
        plt.show()


class AirfoilBenchmark(Benchmark):
    def __init__(self, objs='cl/cd&cl', **kwargs):
        search_space = AirfoilSearchSpace()
        evaluator = AirfoilEvaluator()
        super().__init__(search_space, evaluator, **kwargs)

    @property
    def name(self):
        return 'AirfoilBenchmark'

    def test(self):
        sample = self.search_space.sample(1)
        self.search_space.visualize(sample)
        X = self.search_space.encode(sample)
        F = self.evaluator.evaluate(X)
        # self.evaluator.cl_visualize(F[1])
        print(sample)
        self.search_space.visualize(sample)
        print(X)
        print(F)


if __name__ == '__main__':
    benchmark = AirfoilBenchmark()
    benchmark.test()
