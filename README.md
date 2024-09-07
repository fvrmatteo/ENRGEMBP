# Efficient Normalized Reduction and Generation of Equivalent Multivariate Binary Polynomials

This repository contains a pure Python implementation of the `NormalizePolynomial` and `EquivalentPolynomial` algorithms described in the NDSS2024 paper [Efficient Normalized Reduction and Generation of Equivalent Multivariate Binary Polynomials](https://www.ndss-symposium.org/ndss-paper/auto-draft-436/) for univariate and multivariate binary polynomials. The implementations try to match as close as possible the pseudocode present in the paper, providing support for sparse and dense vectors/matrices and relying on the full or partial Kronecker product implementation. No attempt has been done to optimise the speed of the implementation or parallelise it.

# PyPy

Using [PyPy](https://pypy.org/) to run the script is suggested to get a speed boost until someone will put in the effort to rewrite this code in a more performant way.
