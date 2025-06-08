import cma, numpy as np

# https://cma-es.github.io/apidocs-pycma/ -> api documentation

class MeanCenterCMA(cma.CMAEvolutionStrategy):
    """CMA-ES z średnią TODO(WYTESTUJ)."""

    def tell(self, X, fit, *args, **kwargs):
       
        self.mean = np.median(X, axis=0)  

        super().tell(X, fit, *args, **kwargs)
