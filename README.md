# qiskit-dynamics

[![License](https://img.shields.io/github/license/Qiskit/qiskit-experiments.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

**This repo is still under active development, there will be breaking API changes**

**Qiskit Dynamics** is an open-source project for building, transforming, and solving
models of quantum systems in Qiskit. This package is in a pre-release stage and does not yet have
an official stable version, and as such substantial API changes should be expected.

The goal of this package is to provide tools for flexibly constructing models and simulating
quantum dynamics, with an eye towards classical simulation-heavy applications like optimal
control and model learning. To facilitate this, some of the design elements of this package
are:
- Tools for building both models of quantum systems, such as Hamiltonians and Lindblad dissipators.
- Tools for building models of classical electronics, for simulation of signal processing elements
including IQ mixers and filters.
- Automation of model transformations, such as entering a rotating frame or performing the
rotating wave approximation, which typically need to be performed by hand.
- A unified interface providing modular access to various underlying numerical simulation methods,
enabling advanced users to select the solver most appropriate to their problem.
- Full integration with [JAX](https://github.com/google/jax), enabling just-in-time compilation
to accelerate simulation, as well as automatic differentiation for optimization applications.

As we bring this package towards an official release, we have been 


## Contribution Guidelines

If you'd like to contribute to Qiskit Dynamics, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-experiments/issues) for
tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and use our [Qiskit Slack channel](https://qiskit.slack.com) for discussion and
simple questions.
For questions that are more suited for a forum we use the Qiskit tag in the
[Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Authors and Citation

Qiskit Dynamics is the work of [many people](https://github.com/Qiskit/qiskit-terra/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

## License

[Apache License 2.0](LICENSE.txt)
