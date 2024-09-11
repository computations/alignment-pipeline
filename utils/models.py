from dataclasses import dataclass
import numpy
import pathlib


@dataclass
class DistributionBase:

    def __call__(self):
        if not hasattr(self, '_val'):
            self._val = self._get()
        return self._val

    def lock(self, p):
        self._val = p


@dataclass
class StaticDistribution(DistributionBase):
    param: float

    def _get(self):
        return self.param


@dataclass
class UniformDistribution(DistributionBase):
    start: float
    end: float

    def _get(self):
        return numpy.random.uniform(self.start, self.end)


@dataclass
class GammaRateDistribution(DistributionBase):
    shape: float
    scale: float

    def _get(self):
        return numpy.random.gamma(self.shape, self.scale)


@dataclass
class LogNormalRateDistribution(DistributionBase):
    mu: float
    sigma: float

    def _get(self):
        return numpy.random.lognormal(self.mu, self.sigma)


@dataclass
class InverseGaussianRateDistribution(DistributionBase):
    mu: float
    l: float

    def _get(self):
        return numpy.random.wald(self.mu, self.l)


@dataclass
class BetaDistribution(DistributionBase):
    alpha: float
    beta: float

    def _get(self):
        return numpy.random.beta(self.alpha, self.beta)


class ClampedBetaDistribution(BetaDistribution):
    max: float

    def _get(self):
        while True:
            val = numpy.random.beta(self.alpha, self.beta)
            if val <= self.max:
                return val


Distribution = (StaticDistribution | UniformDistribution
                | GammaRateDistribution | LogNormalRateDistribution
                | InverseGaussianRateDistribution | BetaDistribution
                | ClampedBetaDistribution)


def make_distribution(type, **kwargs) -> Distribution:
    match type:
        case "Static":
            return StaticDistribution(kwargs["value"])
        case "Uniform":
            return UniformDistribution(kwargs['start'], kwargs['end'])
        case "Gamma":
            return GammaRateDistribution(kwargs['k'], kwargs['theta'])
        case "LogNormal":
            return LogNormalRateDistribution(kwargs['mu'], kwargs['sigma'])
        case "InverseGaussian":
            return InverseGaussianRateDistribution(
                kwargs['mu'], kwargs['lambda'])
        case "Beta":
            return BetaDistribution(kwargs['alpha'], kwargs['beta'])
        case "ClampedBeta":
            return ClampedBetaDistribution(kwargs['alpha'], kwargs['beta'],
                                           kwargs['max'])


def make_adna_parameter_set(config):
    params = ADNADamageParameterSet(
        make_distribution(**config['nf']),
        make_distribution(**config['ov']),
        make_distribution(**config['ds']),
        make_distribution(**config['ss']),
    )
    return params


@dataclass
class ADNADamageParameterSet:
    nf: Distribution
    ov: Distribution
    ds: Distribution
    ss: Distribution

    @property
    def dict(self):
        return {
            "nf": self.nf(),
            "ov": self.ov(),
            "ds": self.ds(),
            "ss": self.ss(),
        }

    def load(self, p):
        self.nf.lock(p['nf'])
        self.ov.lock(p['ov'])
        self.ds.lock(p['ds'])
        self.ss.lock(p['ss'])


@dataclass
class PyGargammelConfigParams:
    min_frags: int = 10
    max_frags: int = 100
    min_length: int = 15
    ungap: bool = True
    align: bool = False
    format: bool = True


@dataclass
class PyGargammelFiles:
    gzip: pathlib.Path
    text: pathlib.Path
    output: pathlib.Path
    input: pathlib.Path


@dataclass
class PyGargammelConfig:
    path: pathlib.Path
    params: ADNADamageParameterSet
    config: PyGargammelConfigParams
    files: PyGargammelFiles

    @property
    def command(self):
        return (
            self.path + " " +
            f"--nf {self.params.nf()} " +
            f"--overhang {self.params.ov()} " +
            f"--ds {self.params.ds()} " +
            f"--ss {self.params.ss()} " +
            f"--min-fragments {self.config.min_frags} " +
            f"--max-fragments {self.config.max_frags} " +
            f"--min-length {self.config.min_length} " +
            ("--ungap " if self.config.ungap else "") +
            ("--align " if self.config.align else "") +
            ("--format-taxa-name " if self.config.format else "") +
            f"--log {self.files.gzip} " +
            f"--output {self.files.output} " +
            f"--fasta {self.files.input} " +
            f"&> {self.files.text}"
        )
