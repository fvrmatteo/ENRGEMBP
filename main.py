from typing import List, Set, Tuple, Dict, TypeAlias
from random import randrange

import argparse
import time

# ============================================================================

Vector: TypeAlias = List[int]                   # each 'int' is a coefficient in the Vector
Matrix: TypeAlias = List[Vector]                # each 'Vector' is a row in the Matrix
SparseVector: TypeAlias = List[Tuple[int, int]] # each tuple contains a position and a coefficient

# ============================================================================

class Variable:

  def __init__(self, name, power = 1):
    assert power >= 0
    self.name = name
    self.power = power

  def __mul__(self, other):
    if isinstance(other, int):
      return Monomial(other, [ self ])
    if isinstance(other, Variable):
      if self.name == other.name:
        return Variable(self.name, self.power + other.power)
      return Monomial(1, [ self, other ])
    if isinstance(other, Monomial):
      return Monomial(other.coefficient, [ self ] + other.variables)
    if isinstance(other, Polynomial):
      return Polynomial([ (self * m) for m in other.monomials ])
    raise TypeError(f"Variable: ({type(self)}*{type(other)}) not implemented")

  def __rmul__(self, other):
    return self.__mul__(other)

  def __add__(self, other):
    if isinstance(other, int):
      return Polynomial([ self.as_monomial(), monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial([ self.as_monomial(), other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial([ self.as_monomial(), other ])
    if isinstance(other, Polynomial):
      return Polynomial([ self.as_monomial() ] + other.monomials)
    raise TypeError(f"Variable: ({type(self)}+{type(other)}) not implemented")

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, int):
      return Polynomial([ self.as_monomial(), -1 * monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial([ self.as_monomial(), -1 * other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial([ self.as_monomial(), -1 * other ])
    if isinstance(other, Polynomial):
      return Polynomial([ self.as_monomial() ] + [ -1 * m for m in other.monomials ])
    raise TypeError(f"Variable: ({type(self)}-{type(other)}) not implemented")

  def __rsub__(self, other):
    if isinstance(other, int):
      other = monomial_constant(other)
    return other.__sub__(self)

  def __pow__(self, other):
    if isinstance(other, int):
      return Variable(self.name, self.power * other)
    raise TypeError(f"Variable: ({type(self)}**{type(other)}) not implemented")

  def __eq__(self, other):
    return (self.name == other.name) and (self.power == other.power)

  def __lt__(self, other):
    return self.name < other.name

  def __hash__(self):
    return hash(str(self))

  def __str__(self):
    if self.is_degree_0():
      return "1"
    if self.is_degree_1():
      return self.name
    return f"{self.name}**{self.power}"

  def is_degree_0(self):
    return self.power == 0

  def is_degree_1(self):
    return self.power == 1

  def as_monomial(self):
    return Monomial(1, [ self ])

  def as_polynomial(self):
    return Polynomial([ self.as_monomial() ])

class Monomial:

  def __init__(self, coefficient, variables):
    self.coefficient = coefficient
    # Reduce and lexicographically order the variables
    reduced = {}
    for v in variables:
      if v.name not in reduced:
        reduced[v.name] = Variable(v.name, 0)
      reduced[v.name].power += v.power
    self.variables = sorted([ v for k, v in reduced.items() ])

  def __mul__(self, other):
    if isinstance(other, int):
      return Monomial(self.coefficient * other, self.variables)
    if isinstance(other, Variable):
      return Monomial(self.coefficient, [ other ] + self.variables)
    if isinstance(other, Monomial):
      return Monomial(self.coefficient * other.coefficient, self.variables + other.variables)
    if isinstance(other, Polynomial):
      return Polynomial([ (self * m) for m in other.monomials ])
    raise TypeError(f"Monomial: ({type(self)}*{type(other)}) not implemented")

  def __rmul__(self, other):
    return self.__mul__(other)

  def __add__(self, other):
    if isinstance(other, int):
      return Polynomial([ self, monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial([ self, other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial([ self, other ])
    if isinstance(other, Polynomial):
      return Polynomial([ self ] + other.monomials)
    raise TypeError(f"Monomial: ({type(self)}+{type(other)}) not implemented")

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, int):
      return Polynomial([ self, -1 * monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial([ self, -1 * other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial([ self, -1 * other ])
    if isinstance(other, Polynomial):
      return Polynomial([ self ] + [ -1 * m for m in other.monomials ])
    raise TypeError(f"Monomial: ({type(self)}-{type(other)}) not implemented")

  def __rsub__(self, other):
    if isinstance(other, int):
      other = monomial_constant(other)
    return other.__sub__(self)

  def __pow__(self, other):
    raise TypeError(f"Monomial: ({type(self)}**{type(other)}) not implemented")

  def __eq__(self, other):
    return (self.coefficient == other.coefficient) and (set(self.variables) == set(other.variables))

  def __lt__(self, other):
    # Extract all the variables and powers
    variables = set()
    self_powers = {}
    other_powers = {}
    for v in self.variables:
      self_powers[v.name] = v.power
      variables.add(v.name)
    for v in other.variables:
      other_powers[v.name] = v.power
      variables.add(v.name)
    for n in variables:
      if n not in self_powers:
        self_powers[n] = 0
      if n not in other_powers:
        other_powers[n] = 0
    # Compare the monomials via inverse lexicographic ordering
    for n in sorted(list(variables), reverse=True):
      if self_powers[n] > other_powers[n]:
        return True
      elif self_powers[n] < other_powers[n]:
        return False
    # Same variables and powers
    return True

  def __hash__(self):
    return hash(str(self))

  def __str__(self):
    variables = [ str(v) for v in self.variables ]
    if self.is_constant():
      return f"{self.coefficient}"
    elif self.is_symbolic():
      return f"{"*".join(variables)}"
    return f"{self.coefficient}*{"*".join(variables)}"

  def is_constant(self) -> bool:
    return len(self.variables) == 0

  def is_symbolic(self) -> bool:
    return self.coefficient == 1

  def as_polynomial(self):
    return Polynomial([ self ])

class Polynomial:

  def __init__(self, monomials):
    # Reduce and sort the monomials via inverse lexicographic ordering
    reduced = {}
    for m in monomials:
      v = Monomial(1, m.variables)
      if v not in reduced:
        reduced[v] = 0
      reduced[v] += m.coefficient
    self.monomials = sorted([ Monomial(v, list(k.variables)) for k, v in reduced.items() if (v != 0) ])

  def __mul__(self, other):
    if isinstance(other, int):
      return Polynomial([ (m * other) for m in self.monomials ])
    if isinstance(other, Variable):
      v = other.as_monomial()
      return Polynomial([ (m * v) for m in self.monomials ])
    if isinstance(other, Monomial):
      return Polynomial([ (m * other) for m in self.monomials ])
    if isinstance(other, Polynomial):
      monomials = []
      for x in self.monomials:
        for y in other.monomials:
          monomials.append(x * y)
      return Polynomial(monomials)
    raise TypeError(f"Polynomial: ({type(self)}*{type(other)}) not implemented")

  def __rmul__(self, other):
    return self.__mul__(other)

  def __add__(self, other):
    if isinstance(other, int):
      return Polynomial(self.monomials + [ monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial(self.monomials + [ other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial(self.monomials + [ other ])
    if isinstance(other, Polynomial):
      return Polynomial(self.monomials + other.monomials)
    raise TypeError(f"Polynomial: ({self}+{other}) not implemented")

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, int):
      return Polynomial(self.monomials + [ -1 * monomial_constant(other) ])
    if isinstance(other, Variable):
      return Polynomial(self.monomials + [ -1 * other.as_monomial() ])
    if isinstance(other, Monomial):
      return Polynomial(self.monomials + [ -1 * other ])
    if isinstance(other, Polynomial):
      return Polynomial(self.monomials + [ -1 * m for m in other.monomials ])
    raise TypeError(f"Polynomial: ({self}-{other}) not implemented")

  def __rsub__(self, other):
    if isinstance(other, int):
      other = monomial_constant(other)
    return other.__sub__(self)

  def __pow__(self, other):
    raise TypeError(f"Polynomial: ({self}**{other}) not implemented")

  def __eq__(self, other):
    return self.monomials == other.monomials

  def __hash__(self):
    return hash(str(self))

  def __str__(self):
    if self.is_null():
      return "0"
    return f"{" + ".join([ str(m) for m in self.monomials ])}"

  def is_null(self) -> bool:
    return len(self.monomials) == 0

  def degree(self) -> int:
    d = 0
    for m in self.monomials:
      for v in m.variables:
        if v.power > d:
          d = v.power
    return d

  def __variables(self) -> Set[str]:
    variables = set()
    for m in self.monomials:
      for v in m.variables:
        variables.add(v.name)
    return variables

  def variables_count(self) -> int:
    return len(self.__variables())

  def is_constant(self) -> bool:
    return len(self.__variables()) == 0

  def is_univariate(self) -> bool:
    return len(self.__variables()) == 1

  def is_multivariate(self) -> bool:
    return len(self.__variables()) > 1

  def univariate_coefficient_at(self, d: int) -> int:
    assert self.is_constant() or self.is_univariate()
    for m in self.monomials:
      if m.is_constant():
        if d == 0:
          return m.coefficient
      elif m.variables[0].power == d:
        return m.coefficient
    return 0

  def multivariate_coefficients_dense(self) -> Vector:
    d = self.degree()
    k = self.variables_count()
    coefficients = [0] * (d + 1)**k
    for p, c in self.multivariate_coefficients_sparse():
      coefficients[p] = c
    return coefficients

  def multivariate_coefficients_sparse(self) -> SparseVector:
    variables = sorted(list(self.__variables()))
    degree = self.degree()
    coefficients = []
    for m in self.monomials:
      # Get all the monomial powers
      powers = {}
      for v in m.variables:
        powers[v.name] = v.power
      for n in variables:
        if n not in powers:
          powers[n] = 0
      # Compute the position
      position = 0
      for i in range(len(variables)):
        position += (powers[variables[i]] * (degree + 1)**i)
      # Store the position and coefficient
      coefficients.append((position, m.coefficient))
    return coefficients

  def as_polynomial(self):
    return self

def monomial_constant(c: int) -> Monomial:
  return Monomial(c, [])

def polynomial_constant(c: int) -> Polynomial:
  return Polynomial([ monomial_constant(c) ])

# ============================================================================

def run_polynomial_tests(debug: bool):

  if debug:
    print("="*200)

  x0 = Variable("x0")
  x1 = Variable("x1")
  x2 = Variable("x2")
  x3 = Variable("x3")

  # Variable * int
  v0 = x0 * 2

  # Variable * Variable
  v1 = x0 * x1

  # Variable * Monomial
  v2 = x0 * (x1 * x2)

  # Variable * Polynomial
  v3 = x0 * (x1 + x2)

  # Variable + int
  v4 = x0 + 2

  # Variable + Variable
  v5 = x0 + x1

  # Variable + Monomial
  v6 = x0 + (x1 * 2)

  # Variable + Polynomial
  v7 = x0 + (x1 + x2)

  if debug:
    print(f"v0: {v0}")
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v3: {v3}")
    print(f"v4: {v4}")
    print(f"v5: {v5}")
    print(f"v6: {v6}")
    print(f"v7: {v7}")

  # Monomial * int
  m0 = (x0 * 2) * 2

  # Monomial * Variable
  m1 = (x0 * 2) * x1

  # Monomial * Monomial
  m2 = (x0 * 2) * (x1 * 2)

  # Monomial * Polynomial
  m3 = (x0 * 2) * (x2 + x1)

  # Monomial + int
  m4 = (x0 * 2) + 2

  # Monomial + Variable
  m5 = (x0 * 2) + x1

  # Monomial + Monomial
  m6 = (x0 * 2) + (x1 * 2)

  # Monomial + Polynomial
  m7 = (x0 * 2) + (x1 + x2)

  if debug:
    print(f"m0: {m0}")
    print(f"m1: {m1}")
    print(f"m2: {m2}")
    print(f"m3: {m3}")
    print(f"m4: {m4}")
    print(f"m5: {m5}")
    print(f"m6: {m6}")
    print(f"m7: {m7}")

  # Polynomial * int
  p0 = (x0**2 + x0 + 2) * 2

  # Polynomial * Variable
  p1 = (x0**2 + x0 + 2) * x1

  # Polynomial * Monomial
  p2 = (x0**2 + x0 + 2) * (x1 * 2)

  # Polynomial * Polynomial
  p3 = (x0**2 + x0 + 2) * (x1**2 + x1 + 2)

  # Polynomial + int
  p4 = (x0**2 + x0 + 2) + 2

  # Polynomial + Variable
  p5 = (x0**2 + x0 + 2) + x1

  # Polynomial + Monomial
  p6 = (x0**2 + x0 + 2) + (x1 * 2)

  # Polynomial + Polynomial
  p7 = (x0**2 + x0 + 2) + (x1**2 + x1 + 2)

  if debug:
    print(f"p0: {p0}")
    print(f"p1: {p1}")
    print(f"p2: {p2}")
    print(f"p3: {p3}")
    print(f"p4: {p4}")
    print(f"p5: {p5}")
    print(f"p6: {p6}")
    print(f"p7: {p7}")

  # Variable == Variable
  e0 = (x0 == x1)
  e1 = (x0**1 == x0)
  e2 = (x0**2 == x0)

  # Monomial == Monomial
  e3 = ((x0 * 2) == (x1 * 2))
  e4 = ((x0 * 2) == (2 * x0))
  e5 = (2*(x0**2)*(x1**3)) == (2*(x1*x1)*(x0*x0)*x1)

  # Polynomial == Polynomial
  e6 = ((x0 + x1) == (x3 + x2))
  e7 = ((x0 + x1) == (x1 + x0))

  assert e0 == False
  assert e1 == True
  assert e2 == False
  assert e3 == False
  assert e4 == True
  assert e5 == True
  assert e6 == False
  assert e7 == True

  if debug:
    print(f"e0: {e0}")
    print(f"e1: {e1}")
    print(f"e2: {e2}")
    print(f"e3: {e3}")
    print(f"e4: {e4}")
    print(f"e5: {e5}")
    print(f"e6: {e6}")
    print(f"e7: {e7}")

  # set(Variables)
  s0 = set([x0, x1])
  s1 = set([x0, x0**1])
  s2 = set([x0, x0**2])

  # set(Monomials)
  s3 = set([(x0 * 2), (x1 * 2)])
  s4 = set([(x0 * 2), (2 * x0)])
  s5 = set([(2*(x0**2)*(x1**3)), (2*(x1*x1)*(x0*x0)*x1)])

  # set(Polynomials)
  s6 = set([(x0 + x1), (x3 + x2)])
  s7 = set([(x0 + x1), (x1 + x0)])

  if debug:
    print(f"s0: {sorted([str(p) for p in s0])}")
    print(f"s1: {sorted([str(p) for p in s1])}")
    print(f"s2: {sorted([str(p) for p in s2])}")
    print(f"s3: {sorted([str(p) for p in s3])}")
    print(f"s4: {sorted([str(p) for p in s4])}")
    print(f"s5: {sorted([str(p) for p in s5])}")
    print(f"s6: {sorted([str(p) for p in s6])}")
    print(f"s7: {sorted([str(p) for p in s7])}")

def matrix_by_sparse_vector_mod_m(M: Matrix, v: SparseVector, m: int) -> Vector:
  # TODO: Understand page 9 of the paper
  pass

def matrix_by_dense_vector_mod_m(M: Matrix, v: Vector, m: int) -> Vector:
  assert len(M[0]) == len(v)
  r = len(M)
  c = len(M[0])
  R = [0] * len(M)
  for i in range(r):
    for j in range(c):
      R[i] += (M[i][j] * v[j]) % m
    R[i] %= m
  return R

def matrix_kroneker_product_mod_m(M: Matrix, k: int, m: int) -> Matrix:
  Mr = len(M)
  Mc = len(M[0])
  T = M
  for _ in range(k - 1):
    Tr = len(T)
    Tc = len(T[0])
    N = [[0 for _ in range(Tc*Mc)] for _ in range(Tr*Mr)]
    for i in range(len(N)):
      for j in range(len(N[0])):
        N[i][j] = (T[i // Tr][j // Tc] * M[i % Mr][j % Mc]) % m
    T = N
  return T

def matrix_inverse_mod_m(M: Matrix, m: int) -> Matrix:
  assert len(M) > 0
  assert len(M) == len(M[0])
  assert sum([ M[i][i] for i in range(len(M)) ]) == len(M)
  # The matrices we are processing by design are:
  # - Squared
  # - Upper triangular
  # - With all the diagonal values set to 1
  # So we can use backward-substitution to calculate the inverse modulo 'm'
  n = len(M)
  # Initialize the inverse matrix as an identity matrix
  IM = [[0] * n for _ in range(n)]
  # Copy the diagonal values (all 1)
  for i in range(n):
    IM[i][i] = M[i][i]
  # Backward-substitution for the upper triangular elements
  for i in range(n - 1, -1, -1):
    for j in range(i + 1, n):
      s = 0
      for k in range(i + 1, j + 1):
        s += M[i][k] * IM[k][j]
        s %= m
      IM[i][j] = (-s * IM[i][i]) % m
  # Return the inverse matrix
  return IM

def matrix_extract(M: Matrix, r: int, c: int) -> Matrix:
  E = [[0 for _ in range(c)] for _ in range(r)]
  for i in range(r):
    for j in range(c):
      E[i][j] = M[i][j]
  return E

def reduce_univariate_coefficients(v: Vector, b: int, w: int) -> Vector:
  r = [ a for a in v ]
  for j in range(2, b):
    r[j] %= cj_univariate(j, w)
  return r

def reduce_multivariate_coefficients(v: Vector, d: int, k: int, w: int) -> Vector:
  r = [ a for a in v ]
  for j in range((d + 1)**k):
    if r[j]:
      p = position_to_powers_vector(j, d, k)
      r[j] %= cj_multivariate(p, w)
  return r

def expand_univariate_coefficients(v: Vector, b: int, w: int) -> Vector:
  r = [ a for a in v ]
  for j in range(2, b):
    r[j] += sj(j, w) * cj_univariate(j, w)
  return r

def expand_multivariate_coefficients(v: Vector, b: int, d: int, k: int, w: int) -> Vector:
  r = [ a for a in v ]
  for j in range(2, b**k):
    p = position_to_powers_vector(j, d, k)
    r[j] += sj(j, w) * cj_multivariate(p, w)
  return r

def vector_to_univariate_polynomial(v: Vector) -> Polynomial:
  x = Variable("x")
  p = polynomial_constant(0)
  for i in range(len(v)):
    p += (v[i] * x**i)
  return p

def vector_to_multivariate_polynomial(v: Vector, d: int, k: int) -> Polynomial:
  # Generate the correct amount of variables
  x = {}
  for i in range(k):
    x[i] = Variable(f"x{i}")
  # Translate the vector into a multivariate polynomial
  p = polynomial_constant(0)
  for j in range(len(v)):
    m = v[j]
    powers = position_to_powers_vector(j, d, k)
    for i in range(len(powers)):
      m *= x[i]**powers[i]
    p += m
  # Always return a polynomial
  return p.as_polynomial()

def position_to_powers_vector(j: int, d: int, k: int) -> Vector:
    powers = []
    for i in range(k - 1, 0, -1):
      m = (d + 1)**i
      a = j // m
      b = j % m
      powers.insert(0, a)
      if i == 1:
        powers.insert(0, b)
      else:
        j = b
    return powers

def obtain_multivariate_factorial_coefficients(v: Vector, d: int, k: int) -> Tuple[int, Vector]:
  # Remove the rightmost zeros
  r = 0
  for i in range(len(v)):
    if v[i]:
      r = i
  v = v[:r + 1]
  # Obtain the powers, the coefficient and the maximum degree
  max_degree = 0
  coefficients_and_powers = []
  for j in range(len(v)):
    coefficient = v[j]
    if coefficient:
      powers = position_to_powers_vector(j, d, k)
      coefficients_and_powers.append((coefficient, powers))
      for power in powers:
        if power > max_degree:
          max_degree = power
  # Rewrite the coefficients in the new degree factorial basis
  coefficients = [ 0 ] * ((max_degree + 1)**k)
  for coefficient, powers in coefficients_and_powers:
    # Compute the position
    position = 0
    for i in range(len(powers)):
      position += (powers[i] * (max_degree + 1)**i)
    # Store the coefficient
    coefficients[position] = coefficient
  return max_degree, coefficients

def print_matrix(M: Matrix):
  for R in M:
    print("[", end="")
    for c in R:
      print(f"{c: 4d}", end="")
    print("]")

def v2(m: int) -> int:
  s = 0
  r = 1
  while True:
    e = 2**r
    i = m // e
    r += 1
    s += i
    if i == 0:
      break
  return s

def xj(j: int) -> Polynomial:
  p = polynomial_constant(1)
  x = Variable("x").as_polynomial()
  for i in range(j):
    p *= (x - i)
  return p

def cj_univariate(j: int, w: int) -> int:
  return 2**max(w - v2(j), 0)

def cj_multivariate(j: Vector, w: int) -> int:
  e = w
  for i in j:
    e -= v2(i)
  if e < 0:
    return 1
  return 2**e

def sj(j: int, w: int) -> int:
  return randrange(2**min(v2(j), w) - 1)

def C(d: int, w: int) -> Matrix:
  m = 2**w
  c = [[0 for _ in range(d)] for _ in range(d)]
  for j in range(d):
    x = xj(j)
    for i in range(d):
      a = x.univariate_coefficient_at(i)
      c[i][j] = (m + a if a < 0 else a) % m
  return c

def F(C: Matrix, r: int, c: int, w: int) -> Matrix:
  # Compute the inverse of C
  IC = matrix_inverse_mod_m(C, 2**w)
  # Select the necessary rows and columns
  return matrix_extract(IC, r, c)

def normalize_univariate_polynomial(p: Polynomial, w: int, debug: bool) -> Polynomial:

  dw = w + 2
  d = p.degree()
  v = [ p.univariate_coefficient_at(i) for i in range(d + 1) ]
  l = min(d + 1, dw)

  if debug:
    print(f"dw = {dw}")
    print(f"d = {d}")
    print(f"v = {v}")
    print(f"l = {l}")

  MC = C(d + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MF = F(MC, l, d + 1, w)

  if debug:
    print(f"F[{len(MF)}, {len(MF[0])}]:")
    print_matrix(MF)

  u = matrix_by_dense_vector_mod_m(MF, v, 2**w)

  if debug:
    print(f"u: {u}")

  u = reduce_univariate_coefficients(u, l, w)[:l]

  if debug:
    print(f"u: {u}")

  MC = matrix_extract(MC, l, l)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  v = matrix_by_dense_vector_mod_m(MC, u, 2**w)

  if debug:
    print(f"v: {v}")

  return vector_to_univariate_polynomial(v)

def equivalent_univariate_polynomial(p: Polynomial, w: int, t: int, debug: bool) -> Polynomial:

  assert t >= p.degree()

  d = p.degree()
  v = [ p.univariate_coefficient_at(i) for i in range(d + 1) ]

  if debug:
    print(f"δ = {t}")
    print(f"d = {d}")
    print(f"v = {v}")

  MC = C(t + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MF = F(MC, t + 1, d + 1, w)

  if debug:
    print(f"F[{len(MF)}, {len(MF[0])}]:")
    print_matrix(MF)

  u = matrix_by_dense_vector_mod_m(MF, v, 2**w)

  if debug:
    print(f"u: {u}")

  u = expand_univariate_coefficients(u, t + 1, w)

  if debug:
    print(f"u: {u}")

  MC = matrix_extract(MC, t + 1, t + 1)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  v = matrix_by_dense_vector_mod_m(MC, u, 2**w)

  if debug:
    print(f"v: {v}")

  return vector_to_univariate_polynomial(v)

def normalize_multivariate_polynomial(p: Polynomial, w: int, debug: bool) -> Polynomial:

  d = p.degree()
  k = p.variables_count()
  vs = p.multivariate_coefficients_sparse()
  vd = p.multivariate_coefficients_dense()

  if debug:
    print(f"d: {d}")
    print(f"k: {k}")
    print(f"vs: {vs}")
    print(f"vd: {vd}")

  MC = C(d + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MF = F(MC, d + 1, d + 1, w)

  if debug:
    print(f"F[{len(MF)}, {len(MF[0])}]:")
    print_matrix(MF)

  MFK = matrix_kroneker_product_mod_m(MF, k, 2**w)

  if debug:
    print(f"FK[{len(MFK)}, {len(MFK[0])}]:")
    print_matrix(MFK)

  ud = matrix_by_dense_vector_mod_m(MFK, vd, 2**w)

  if debug:
    print(f"ud: {ud}")

  # us = matrix_by_sparse_vector_mod_m(MF, vs, 2**w)

  # if debug:
  #   print(f"us: {us}")

  ud = reduce_multivariate_coefficients(ud, d, k, w)

  if debug:
    print(f"ud: {ud}")

  # us = reduce_multivariate_coefficients(us, d, k, w)

  # if debug:
  #   print(f"us: {us}")

  dw, ud = obtain_multivariate_factorial_coefficients(ud, d, k)

  if debug:
    print(f"dw: {dw}")
    print(f"ud: {ud}")

  MC = C(dw + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MCK = matrix_kroneker_product_mod_m(MC, k, 2**w)

  if debug:
    print(f"CK[{len(MCK)}, {len(MCK[0])}]:")
    print_matrix(MCK)

  vd = matrix_by_dense_vector_mod_m(MCK, ud, 2**w)

  if debug:
    print(f"vd: {vd}")

  # vs = matrix_by_sparse_vector_mod_m(MC, us, 2**w)

  # if debug:
  #   print(f"vs: {vs}")

  return vector_to_multivariate_polynomial(vd, dw, k)

def equivalent_multivariate_polynomial(p: Polynomial, w: int, t: int, debug: bool) -> Polynomial:

  assert t >= p.degree()

  d = p.degree()
  k = p.variables_count()
  vs = p.multivariate_coefficients_sparse()
  vd = p.multivariate_coefficients_dense()

  if debug:
    print(f"δ: {t}")
    print(f"d: {d}")
    print(f"k: {k}")
    print(f"vs: {vs}")
    print(f"vd: {vd}")

  MC = C(t + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MF = F(MC, t + 1, d + 1, w)

  if debug:
    print(f"F[{len(MF)}, {len(MF[0])}]:")
    print_matrix(MF)

  MFK = matrix_kroneker_product_mod_m(MF, k, 2**w)

  if debug:
    print(f"FK[{len(MFK)}, {len(MFK[0])}]:")
    print_matrix(MFK)

  ud = matrix_by_dense_vector_mod_m(MFK, vd, 2**w)

  if debug:
    print(f"ud: {ud}")

  ud = expand_multivariate_coefficients(ud, t + 1, t, k, w)

  if debug:
    print(f"ud: {ud}")

  dw, ud = obtain_multivariate_factorial_coefficients(ud, t, k)

  if debug:
    print(f"dw: {dw}")
    print(f"ud: {ud}")

  MC = C(dw + 1, w)

  if debug:
    print(f"C[{len(MC)}, {len(MC[0])}]:")
    print_matrix(MC)

  MCK = matrix_kroneker_product_mod_m(MC, k, 2**w)

  if debug:
    print(f"CK[{len(MCK)}, {len(MCK[0])}]:")
    print_matrix(MCK)

  vd = matrix_by_dense_vector_mod_m(MCK, ud, 2**w)

  if debug:
    print(f"vd: {vd}")

  return vector_to_multivariate_polynomial(vd, dw, k)

# ============================================================================

x = Variable("x")
x0 = Variable("x0")
x1 = Variable("x1")

univariate_polynomials = [
  (8, x**10),
  (8, 248*x**2 + 97*x),
  (8, 140*x**14 + 91*x**13 + 188*x**12 + 170*x**11 + 130*x**10 + 174*x**9 + 176*x**8 + 132*x**7 + 19*x**6 + 160*x**5 + 143*x**4 + 67*x**3 + 112*x**2 + 193*x),
  (8, 237*x + 64*x**2 + 40*x**3 + 54*x**5 + 196*x**6 + 236*x**7 + 123*x**8 + 230*x**9 + 88*x**10 + 248*x**11 + 22*x**12 + 32*x**13 + 108*x**14 + 184*x**15 + 39*x**16),
  (64, 13351459109906670064*x + 3458764513820540928*x**2 + 13835058055282163712*x**3 + 13907115649320091648*x**4 + 17185736178045812736*x**5 + 15294224334550204416*x**6 + 13249590103723999232*x**7 + 6422133068630327296*x**8 + 4278419646001971200*x**9 + 7430939385161318400*x**10 + 9196350439090552832*x**11 + 6368089873101881344*x**12 + 16041821872693706752*x**13 + 2458965396544290816*x**14)
]

multivariate_polynomials = [
  (8, x0**10*x1),
  (8, 2*x0**10*x1**8 - 56*x0**10*x1**7 - 124*x0**10*x1**6 - 80*x0**10*x1**5 - 30*x0**10*x1**4 + 104*x0**10*x1**3 + 24*x0**10*x1**2 - 96*x0**10*x1 - 90*x0**9*x1**8 - 40*x0**9*x1**7 - 52*x0**9*x1**6 + 16*x0**9*x1**5 + 70*x0**9*x1**4 - 72*x0**9*x1**3 - 56*x0**9*x1**2 - 32*x0**9*x1 - 52*x0**8*x1**8 - 80*x0**8*x1**7 - 104*x0**8*x1**6 + 32*x0**8*x1**5 + 12*x0**8*x1**4 + 112*x0**8*x1**3 - 112*x0**8*x1**2 - 64*x0**8*x1 + 44*x0**7*x1**8 + 48*x0**7*x1**7 + 88*x0**7*x1**6 + 32*x0**7*x1**5 + 108*x0**7*x1**4 - 16*x0**7*x1**3 + 16*x0**7*x1**2 - 64*x0**7*x1 + 82*x0**6*x1**8 + 8*x0**6*x1**7 + 36*x0**6*x1**6 + 48*x0**6*x1**5 + 50*x0**6*x1**4 - 88*x0**6*x1**3 - 40*x0**6*x1**2 - 96*x0**6*x1 - 26*x0**5*x1**8 - 40*x0**5*x1**7 + 76*x0**5*x1**6 + 16*x0**5*x1**5 - 122*x0**5*x1**4 - 72*x0**5*x1**3 - 56*x0**5*x1**2 - 32*x0**5*x1 - 64*x0**4*x1**8 + 128*x0**4*x1**6 - 64*x0**4*x1**4 + 72*x0**3*x1**8 + 32*x0**3*x1**7 - 112*x0**3*x1**6 - 64*x0**3*x1**5 - 56*x0**3*x1**4 - 96*x0**3*x1**3 + 96*x0**3*x1**2 + 128*x0**3*x1 + 32*x0**2*x1**8 + 128*x0**2*x1**7 + 64*x0**2*x1**6 + 32*x0**2*x1**4 + 128*x0**2*x1**3 + 128*x0**2*x1**2),
  (8, x0**3*x1**8 + 228*x0**3*x1**7 + 253*x0**2*x1**8 + 66*x0**3*x1**6 + 84*x0**2*x1**7 + 2*x0*x1**8 + 4*x0**4*x1**4 + 88*x0**3*x1**5 + 58*x0**2*x1**6 + 200*x0*x1**7 + 232*x0**4*x1**3 + 89*x0**3*x1**4 + 248*x0**2*x1**5 + 132*x0*x1**6 + 44*x0**4*x1**2 + 68*x0**3*x1**3 + 217*x0**2*x1**4 + 176*x0*x1**5 + 232*x0**4*x1 + 4*x0**3*x1**2 + 220*x0**2*x1**3 + 202*x0*x1**4 + 224*x0**3*x1 + 193*x0**2*x1**2 + 248*x0*x1**3 + 8*x0**2*x1 + 16*x0*x1**2 + 48*x0*x1),
  (64, 14411518807585587200*x1 + 14411518807585587201*x0 + 11853474219239145472*x1**2 + 3819052484010180608*x0*x1 + 1188950301625810944*x0**2 + 14740281580383633408*x1**3 + 2049137830453575680*x0*x1**2 + 8588364489395535872*x0**2*x1 + 12056136202470817792*x0**3 + 6506012611690102784*x1**4 + 11240984669916758016*x0*x1**3 + 5225301467656617984*x0**2*x1**2 + 2751699372323373056*x0**3*x1 + 2261369962893410304*x0**4 + 18413600395201871872*x1**5 + 9593300524996755456*x0*x1**4 + 18404663564691308544*x0**2*x1**3 + 15769494832726147072*x0**3*x1**2 + 14863919463903789056*x0**4*x1 + 7872643992364515328*x0**5 + 13846141132490145792*x1**6 + 16782945486372864*x0*x1**5 + 17005732930439348224*x0**2*x1**4 + 12736531589924651008*x0**3*x1**3 + 5352070760292679680*x0**4*x1**2 + 12829946097820499968*x0**5*x1 + 7826059883718901760*x0**6 + 6178701194240720896*x1**7 + 8679887837291610112*x0*x1**6 + 5320378437133664256*x0**2*x1**5 + 11201024019316867072*x0**3*x1**4 + 4283758874466451456*x0**4*x1**3 + 16850068472225333248*x0**5*x1**2 + 10985994729295970304*x0**6*x1 + 8484596980012548096*x0**7 + 699484833457373184*x1**8 + 14863930459020066816*x0*x1**7 + 17598032147740884992*x0**2*x1**6 + 17084244657732321280*x0**3*x1**5 + 16023207690591272960*x0**4*x1**4 + 3952284705994309632*x0**5*x1**3 + 4466072196002873344*x0**6*x1**2 + 1330200162399682560*x0**7*x1 + 16890607191063527424*x0**8 + 17054244276810022912*x1**9 + 6963473151799853056*x0*x1**8 + 16693408131354460160*x0**2*x1**7 + 7820810540329992192*x0**3*x1**6 + 3518346361734955008*x0**4*x1**5 + 6118204002896183296*x0**5*x1**4 + 13020525822652448768*x0**6*x1**3 + 7231244708935106560*x0**7*x1**2 + 17486774009235243008*x0**8*x1 + 2746078050406891520*x0**9 + 3850611336765505536*x1**10 + 8747247561741434880*x0*x1**9 + 16477924760211161088*x0**2*x1**8 + 2294685165215023104*x0**3*x1**7 + 15776895473694015488*x0**4*x1**6 + 7334151713030406144*x0**5*x1**5 + 16525566066466750464*x0**6*x1**4 + 11056056159973670912*x0**7*x1**3 + 15641588421728665600*x0**8*x1**2 + 13281933406486134784*x0**9*x1 + 11334976094999674880*x0**10 + 9148114473137995776*x1**11 + 14171455841118453760*x0*x1**10 + 16030266001190813696*x0**2*x1**9 + 1424845174130868224*x0**3*x1**8 + 16020749740937314304*x0**4*x1**7 + 12883940751628566528*x0**5*x1**6 + 4869030348450693120*x0**6*x1**5 + 12022187501062979584*x0**7*x1**4 + 4241624665472106496*x0**8*x1**3 + 127801961687810048*x0**9*x1**2 + 13771599617131020288*x0**10*x1 + 5004409542981713920*x0**11 + 14727422045646225408*x1**12 + 1372109501938794496*x0*x1**11 + 4239347009955299328*x0**2*x1**10 + 15673572783336456192*x0**3*x1**9 + 7058907278946074624*x0**4*x1**8 + 15092750996402601984*x0**5*x1**7 + 14560149157340774400*x0**6*x1**6 + 14524428648779350016*x0**7*x1**5 + 1736818325989621760*x0**8*x1**4 + 12125513481365487616*x0**9*x1**3 + 10198820918650732544*x0**10*x1**2 + 1049558656305070080*x0**11*x1 + 7316469331599032320*x0**12 + 573298520086282240*x1**13 + 12250191260488302592*x0*x1**12 + 14625347427150856192*x0**2*x1**11 + 14391665241266061312*x0**3*x1**10 + 1342700878657748992*x0**4*x1**9 + 11853825166117175296*x0**5*x1**8 + 2349334899246235648*x0**6*x1**7 + 3438023104209616896*x0**7*x1**6 + 9691687608857395200*x0**8*x1**5 + 14371402015328698368*x0**9*x1**4 + 289514752298713088*x0**10*x1**3 + 4170371103807504384*x0**11*x1**2 + 4654771913522937856*x0**12*x1 + 9944666256021913600*x0**13 + 7083730303688114176*x1**14 + 16733593323918327808*x0*x1**13 + 11473609321103753216*x0**2*x1**12 + 18211685565017882624*x0**3*x1**11 + 15756058169878511616*x0**4*x1**10 + 4957768094912086016*x0**5*x1**9 + 5387565708244680704*x0**6*x1**8 + 4831345211252670464*x0**7*x1**7 + 13179608234027122688*x0**8*x1**6 + 6123786092433899520*x0**9*x1**5 + 5101356621951401984*x0**10*x1**4 + 8573831705729695744*x0**11*x1**3 + 17685152026930970624*x0**12*x1**2 + 1278270003270909952*x0**13*x1 + 3489943684420468736*x0**14 + 17191029007156510720*x1**15 + 12326283758046019584*x0*x1**14 + 12439102442921000960*x0**2*x1**13 + 15886813676540264448*x0**3*x1**12 + 16314113747398426624*x0**4*x1**11 + 8216146242837676032*x0**5*x1**10 + 11016157333397962752*x0**6*x1**9 + 2348004848789618688*x0**7*x1**8 + 7523422739265224704*x0**8*x1**7 + 10025688692296777728*x0**9*x1**6 + 11837363620163878912*x0**10*x1**5 + 381402823734067200*x0**11*x1**4 + 10376133571663364096*x0**12*x1**3 + 13427147782473383936*x0**13*x1**2 + 6683570064886595584*x0**14*x1 + 12322660628425605120*x0**15 + 11240761601881538560*x1**16 + 3138875495752400896*x0*x1**15 + 2084947370792976384*x0**2*x1**14 + 9041842024315092992*x0**3*x1**13 + 8642831079973060608*x0**4*x1**12 + 17737455718098796544*x0**5*x1**11 + 16788261883084800000*x0**6*x1**10 + 10545227918143913984*x0**7*x1**9 + 15051393154546663424*x0**8*x1**8 + 15712333078414229504*x0**9*x1**7 + 16528219327518736384*x0**10*x1**6 + 8982593388503105536*x0**11*x1**5 + 9991984560579018752*x0**12*x1**4 + 8984286456860114944*x0**13*x1**3 + 17700468908760236032*x0**14*x1**2 + 9961868580326735872*x0**15*x1 + 4522358846986911744*x0**16 + 2910916329382871040*x1**17 + 7745881650986221568*x0*x1**16 + 14390996342480568320*x0**2*x1**15 + 306784505123831808*x0**3*x1**14 + 16577156169554984960*x0**4*x1**13 + 6963286962476154880*x0**5*x1**12 + 14761001463610081280*x0**6*x1**11 + 5962701107804241920*x0**7*x1**10 + 175640498143232000*x0**8*x1**9 + 9029967117667008512*x0**9*x1**8 + 16058569437590061056*x0**10*x1**7 + 5161600463300198400*x0**11*x1**6 + 18330162365779148800*x0**12*x1**5 + 11419589390482014208*x0**13*x1**4 + 26945313549844480*x0**14*x1**3 + 1228712351688032256*x0**15*x1**2 + 16535556514132262912*x0**16*x1 + 15054932737276248064*x0**17 + 10160345130800775168*x1**18 + 15354692671383797760*x0*x1**17 + 7316575236682416128*x0**2*x1**16 + 7171187910534758400*x0**3*x1**15 + 15230320197240881152*x0**4*x1**14 + 18240271535992471552*x0**5*x1**13 + 10479029705322266624*x0**6*x1**12 + 18209615781673566208*x0**7*x1**11 + 7166159812059725824*x0**8*x1**10 + 12919972960810303488*x0**9*x1**9 + 3599134779204698112*x0**10*x1**8 + 1943998323144785920*x0**11*x1**7 + 3011021069420068864*x0**12*x1**6 + 458034204957999104*x0**13*x1**5 + 16287439958728245248*x0**14*x1**4 + 1130928009973334016*x0**15*x1**3 + 2182971298865479680*x0**16*x1**2 + 14490371484064677888*x0**17*x1 + 14486075694211137536*x0**18 + 12787671225064226816*x1**19 + 2190142955290886144*x0*x1**18 + 13347021224181497856*x0**2*x1**17 + 6860811076165959680*x0**3*x1**16 + 9335423661118586880*x0**4*x1**15 + 8054331196735225856*x0**5*x1**14 + 125318354767446016*x0**6*x1**13 + 17702398327443685376*x0**7*x1**12 + 8952547268639064064*x0**8*x1**11 + 14710208946503680000*x0**9*x1**10 + 7946392884631044096*x0**10*x1**9 + 10843052204867190784*x0**11*x1**8 + 13942275052612354048*x0**12*x1**7 + 596544584834613248*x0**13*x1**6 + 2796433551459876864*x0**14*x1**5 + 15802865300686241792*x0**15*x1**4 + 13258869007491006464*x0**16*x1**3 + 9599729647951544320*x0**17*x1**2 + 8919751117985808384*x0**18*x1 + 5051619107471884288*x0**19 + 10846064124464463872*x1**20 + 11101826043548794880*x0*x1**19 + 15204293852320497664*x0**2*x1**18 + 9366557622648963072*x0**3*x1**17 + 54400998312509440*x0**4*x1**16 + 12771075269437947904*x0**5*x1**15 + 12822594437244780544*x0**6*x1**14 + 8341446652168503296*x0**7*x1**13 + 5686775165770268672*x0**8*x1**12 + 11944563923819167744*x0**9*x1**11 + 15379194517052719104*x0**10*x1**10 + 10488099409648156672*x0**11*x1**9 + 3727258375882276864*x0**12*x1**8 + 5460112350245814272*x0**13*x1**7 + 15008987762791546880*x0**14*x1**6 + 11234169749826961408*x0**15*x1**5 + 3652898283030839296*x0**16*x1**4 + 17257603882572840960*x0**17*x1**3 + 14468417896603713536*x0**18*x1**2 + 14320324964756488192*x0**19*x1 + 15434894455200022528*x0**20 + 8483905407057395712*x1**21 + 284307498680451072*x0*x1**20 + 1480137369212420096*x0**2*x1**19 + 978129637296046080*x0**3*x1**18 + 4092245147053719552*x0**4*x1**17 + 9504443164338683904*x0**5*x1**16 + 10103337904005709824*x0**6*x1**15 + 6419093877092777984*x0**7*x1**14 + 3309713652113211392*x0**8*x1**13 + 8509881754231242752*x0**9*x1**12 + 6341890388548124672*x0**10*x1**11 + 13689369668804935680*x0**11*x1**10 + 10386398119442513920*x0**12*x1**9 + 17080354651734802432*x0**13*x1**8 + 7239457120423247872*x0**14*x1**7 + 9817333834394370048*x0**15*x1**6 + 12386266116477681664*x0**16*x1**5 + 18015522229196161024*x0**17*x1**4 + 12940369897320349696*x0**18*x1**3 + 1279256228426416128*x0**19*x1**2 + 7761197193035513856*x0**20*x1 + 17393969844203552768*x0**21 + 2916249707658608640*x1**22 + 14369729676475760640*x0*x1**21 + 12036451681624260608*x0**2*x1**20 + 13562391070801920000*x0**3*x1**19 + 11331350053572837376*x0**4*x1**18 + 1961303100970631168*x0**5*x1**17 + 16113366134600105984*x0**6*x1**16 + 16446888251550072832*x0**7*x1**15 + 5952782058751787008*x0**8*x1**14 + 7610611553556234240*x0**9*x1**13 + 1511739045403426816*x0**10*x1**12 + 4538790405557714944*x0**11*x1**11 + 3544700381515743232*x0**12*x1**10 + 96931320349065216*x0**13*x1**9 + 4714445236852228096*x0**14*x1**8 + 148369973883961344*x0**15*x1**7 + 971072716894633984*x0**16*x1**6 + 9617474171719254016*x0**17*x1**5 + 16000536273240850432*x0**18*x1**4 + 2306661802044555264*x0**19*x1**3 + 3248003607523491840*x0**20*x1**2 + 9246943489621622784*x0**21*x1 + 1565397918892818432*x0**22 + 7207538721607974912*x1**23 + 15542286175586025472*x0*x1**22 + 15963449893531877376*x0**2*x1**21 + 8061865925312970752*x0**3*x1**20 + 12872327613939122176*x0**4*x1**19 + 5864171760213491712*x0**5*x1**18 + 15526254786517991424*x0**6*x1**17 + 562915580693774336*x0**7*x1**16 + 18017301269090795520*x0**8*x1**15 + 8027745343198330880*x0**9*x1**14 + 8489403603716800512*x0**10*x1**13 + 5700922975241240576*x0**11*x1**12 + 9652502869051834368*x0**12*x1**11 + 13674810412002836480*x0**13*x1**10 + 3677437557106933760*x0**14*x1**9 + 10326916742724583424*x0**15*x1**8 + 12450978878210768896*x0**16*x1**7 + 765482843436023808*x0**17*x1**6 + 684329754337411072*x0**18*x1**5 + 15893418862011940864*x0**19*x1**4 + 12332953063477149696*x0**20*x1**3 + 10055808245689942016*x0**21*x1**2 + 2572825160313274368*x0**22*x1 + 6524214231081222144*x0**23 + 5623971104698597376*x1**24 + 8784369549544882176*x0*x1**23 + 17335267718844153856*x0**2*x1**22 + 17077534322711822336*x0**3*x1**21 + 1837018643216236544*x0**4*x1**20 + 8792164203412226048*x0**5*x1**19 + 14298048442213826560*x0**6*x1**18 + 3875082228075102208*x0**7*x1**17 + 4715537979264892928*x0**8*x1**16 + 15895901638780452864*x0**9*x1**15 + 436891960773771264*x0**10*x1**14 + 9936154383330508800*x0**11*x1**13 + 16467471656905146368*x0**12*x1**12 + 5958045544059568128*x0**13*x1**11 + 2991884613282365440*x0**14*x1**10 + 10289245866586865664*x0**15*x1**9 + 15921582600004829184*x0**16*x1**8 + 12666384878874886144*x0**17*x1**7 + 8844355706612252672*x0**18*x1**6 + 10724477156274307072*x0**19*x1**5 + 7422684856125390848*x0**20*x1**4 + 18367510382064992256*x0**21*x1**3 + 17771951221513748480*x0**22*x1**2 + 14508081269674246144*x0**23*x1 + 4761540846225096704*x0**24 + 10253268331913494528*x1**25 + 6711476675300016128*x0*x1**24 + 12097958848317587456*x0**2*x1**23 + 5820665881334153216*x0**3*x1**22 + 2741704587974737920*x0**4*x1**21 + 17397984782957346816*x0**5*x1**20 + 14460273801899704320*x0**6*x1**19 + 1810395011528622080*x0**7*x1**18 + 15163322234552467456*x0**8*x1**17 + 17308940104991293440*x0**9*x1**16 + 13274734881744289792*x0**10*x1**15 + 16388522399122653184*x0**11*x1**14 + 4277739192107663360*x0**12*x1**13 + 1501951007497388032*x0**13*x1**12 + 3530727297689911296*x0**14*x1**11 + 3780874465246904320*x0**15*x1**10 + 12939787902226513920*x0**16*x1**9 + 4369745374206312448*x0**17*x1**8 + 5570680111511470080*x0**18*x1**7 + 6389851126640050176*x0**19*x1**6 + 11367999303531167744*x0**20*x1**5 + 17762050185948626944*x0**21*x1**4 + 16272225404263366656*x0**22*x1**3 + 2510703500437454848*x0**23*x1**2 + 15491338215136477184*x0**24*x1 + 17241855523948478464*x0**25 + 2695222186124394496*x1**26 + 960075514239606784*x0*x1**25 + 6970974996022861824*x0**2*x1**24 + 2231997105137549312*x0**3*x1**23 + 16043242435850010624*x0**4*x1**22 + 171626645908733952*x0**5*x1**21 + 13356197167686828032*x0**6*x1**20 + 5818575271415545856*x0**7*x1**19 + 17448938436094935040*x0**8*x1**18 + 6813394657518133248*x0**9*x1**17 + 11833930008979922944*x0**10*x1**16 + 12837687282587729920*x0**11*x1**15 + 342954103577378816*x0**12*x1**14 + 3268610417928994816*x0**13*x1**13 + 1487453953768652800*x0**14*x1**12 + 14817393240606310400*x0**15*x1**11 + 4907374768650240000*x0**16*x1**10 + 15676979849143427072*x0**17*x1**9 + 16805008194062475264*x0**18*x1**8 + 11383345666248048640*x0**19*x1**7 + 12966895599338782720*x0**20*x1**6 + 4109500447456083968*x0**21*x1**5 + 3100994563978117120*x0**22*x1**4 + 9952022112593674240*x0**23*x1**3 + 15045241002122002432*x0**24*x1**2 + 11811405496551104512*x0**25*x1 + 13741674438497443840*x0**26 + 8084050047486447616*x1**27 + 3263906862018940928*x0*x1**26 + 14237524836788768768*x0**2*x1**25 + 5182943218158014464*x0**3*x1**24 + 3136507639669317632*x0**4*x1**23 + 4275040324289748992*x0**5*x1**22 + 10356402093492002816*x0**6*x1**21 + 2837695748166574080*x0**7*x1**20 + 4759964002420355072*x0**8*x1**19 + 8587192804226953216*x0**9*x1**18 + 3938586251910918144*x0**10*x1**17 + 11208685364472164352*x0**11*x1**16 + 5739286190968225792*x0**12*x1**15 + 4097210274606825472*x0**13*x1**14 + 8790757030784745472*x0**14*x1**13 + 1194183692793364480*x0**15*x1**12 + 5161671247006445568*x0**16*x1**11 + 8679256183690375168*x0**17*x1**10 + 726691021767962624*x0**18*x1**9 + 16550169546697633792*x0**19*x1**8 + 14690749481347506176*x0**20*x1**7 + 7391637653761941504*x0**21*x1**6 + 5234325949543407616*x0**22*x1**5 + 17243402407279058944*x0**23*x1**4 + 9981723476203474944*x0**24*x1**3 + 17100877164901142528*x0**25*x1**2 + 2103701627164184576*x0**26*x1 + 7776483725200576512*x0**27 + 14174205099948191744*x1**28 + 7367406697809082368*x0*x1**27 + 1211392859892017152*x0**2*x1**26 + 1937856103472646144*x0**3*x1**25 + 14874463754351314944*x0**4*x1**24 + 4012845800700088320*x0**5*x1**23 + 8382199047129346048*x0**6*x1**22 + 15731053144532332544*x0**7*x1**21 + 14323053911801462784*x0**8*x1**20 + 7792521124117030912*x0**9*x1**19 + 8953938872575490048*x0**10*x1**18 + 1806291240406341632*x0**11*x1**17 + 16503183937119090688*x0**12*x1**16 + 6763578099981508608*x0**13*x1**15 + 18382858144304078848*x0**14*x1**14 + 5964871466219282432*x0**15*x1**13 + 3941202447062396928*x0**16*x1**12 + 16100814187346348032*x0**17*x1**11 + 17839560625212184576*x0**18*x1**10 + 3581425838061942784*x0**19*x1**9 + 11625149179718957056*x0**20*x1**8 + 3196557031365349376*x0**21*x1**7 + 10763914834055901184*x0**22*x1**6 + 12036384749250654208*x0**23*x1**5 + 16896860587288551424*x0**24*x1**4 + 308637970447472640*x0**25*x1**3 + 11930419840463292416*x0**26*x1**2 + 4576169628576999424*x0**27*x1 + 13607730309912561664*x0**28 + 1625384093415617536*x1**29 + 17185805533354732544*x0*x1**28 + 14292165374193811456*x0**2*x1**27 + 2916295770605060096*x0**3*x1**26 + 8774813316175291392*x0**4*x1**25 + 1738555743634979840*x0**5*x1**24 + 2614391729401978880*x0**6*x1**23 + 17326540279652818944*x0**7*x1**22 + 11008513155310208000*x0**8*x1**21 + 4524122793818299392*x0**9*x1**20 + 15230321802000457728*x0**10*x1**19 + 4404653135755730944*x0**11*x1**18 + 11943809240982238208*x0**12*x1**17 + 3855132066753555456*x0**13*x1**16 + 1947450217763684352*x0**14*x1**15 + 4365616296781496320*x0**15*x1**14 + 15499596543730392064*x0**16*x1**13 + 4478000128205947904*x0**17*x1**12 + 17568576795392704512*x0**18*x1**11 + 13553122494879055872*x0**19*x1**10 + 14323844976988236800*x0**20*x1**9 + 6739668410307900416*x0**21*x1**8 + 5527801650736324608*x0**22*x1**7 + 14566330566979133440*x0**23*x1**6 + 7988899109339032576*x0**24*x1**5 + 15284159666596721664*x0**25*x1**4 + 4867921534781415424*x0**26*x1**3 + 15103248014228807680*x0**27*x1**2 + 18012560327794506752*x0**28*x1 + 6738054892742108160*x0**29 + 7507485244350228480*x1**30 + 12446555115872042496*x0*x1**29 + 16020842162914551296*x0**2*x1**28 + 4106334687607993344*x0**3*x1**27 + 15682819158916828160*x0**4*x1**26 + 18005683546750030336*x0**5*x1**25 + 14989074449513974272*x0**6*x1**24 + 939957834199148544*x0**7*x1**23 + 4495033614425935872*x0**8*x1**22 + 10931249797612019200*x0**9*x1**21 + 15957019034717832704*x0**10*x1**20 + 5702973448342645760*x0**11*x1**19 + 5922822011285963776*x0**12*x1**18 + 13507937362016788992*x0**13*x1**17 + 5884016963989805568*x0**14*x1**16 + 7601665738037899264*x0**15*x1**15 + 1064745073347700736*x0**16*x1**14 + 3737549298512685568*x0**17*x1**13 + 15177933438250525184*x0**18*x1**12 + 1028830361367170048*x0**19*x1**11 + 3248464511134667776*x0**20*x1**10 + 16509296581676148224*x0**21*x1**9 + 12570294708040060416*x0**22*x1**8 + 6329929672177526784*x0**23*x1**7 + 14039378142941880320*x0**24*x1**6 + 8071563221673163264*x0**25*x1**5 + 12751845772592853504*x0**26*x1**4 + 3883724450893353984*x0**27*x1**3 + 3556409063747705856*x0**28*x1**2 + 12272347898021930496*x0**29*x1 + 13100903812009135616*x0**30 + 5560396022540590080*x1**31 + 10721478037323235328*x0*x1**30 + 6456621849415826944*x0**2*x1**29 + 16535648382774749696*x0**3*x1**28 + 4270247952108619776*x0**4*x1**27 + 8336050042552667136*x0**5*x1**26 + 4427204219439974912*x0**6*x1**25 + 16335935105336768000*x0**7*x1**24 + 9460771836280562688*x0**8*x1**23 + 1450459324830504960*x0**9*x1**22 + 8340178436289007104*x0**10*x1**21 + 9933396910918396416*x0**11*x1**20 + 7558042966262120448*x0**12*x1**19 + 8345163962147705856*x0**13*x1**18 + 15673470600616159744*x0**14*x1**17 + 4690946930230445568*x0**15*x1**16 + 12271894933672190976*x0**16*x1**15 + 16032387024451932160*x0**17*x1**14 + 17033778032955230720*x0**18*x1**13 + 2129939080500858368*x0**19*x1**12 + 14786883236267452416*x0**20*x1**11 + 7106260179321598976*x0**21*x1**10 + 1662543799095700992*x0**22*x1**9 + 1857885977068335616*x0**23*x1**8 + 13856252199518362624*x0**24*x1**7 + 6169028891854493696*x0**25*x1**6 + 1848073408252094976*x0**26*x1**5 + 13066605153754095104*x0**27*x1**4 + 646807441108799488*x0**28*x1**3 + 11826286159216503808*x0**29*x1**2 + 15550613954451971584*x0**30*x1 + 4866309725992374784*x0**31 + 5472244308788289536*x1**32 + 8942512417428883968*x0*x1**31 + 11710564771204082176*x0**2*x1**30 + 10029437866092233216*x0**3*x1**29 + 2009895843707826688*x0**4*x1**28 + 12455604379469744640*x0**5*x1**27 + 18026082606371637760*x0**6*x1**26 + 6084058355745079808*x0**7*x1**25 + 1043571596453483008*x0**8*x1**24 + 7954609631108166144*x0**9*x1**23 + 16945586507668296192*x0**10*x1**22 + 14390318858902342144*x0**11*x1**21 + 1977217695795788288*x0**12*x1**20 + 3606169586276372992*x0**13*x1**19 + 17034184459209888256*x0**14*x1**18 + 8638400899893701120*x0**15*x1**17 + 4435859463182694912*x0**16*x1**16 + 15245069073356971520*x0**17*x1**15 + 12766680215143120384*x0**18*x1**14 + 12492526466087650816*x0**19*x1**13 + 12164262306564052480*x0**20*x1**12 + 6255598023891574272*x0**21*x1**11 + 10425801303480327680*x0**22*x1**10 + 11685287840283657728*x0**23*x1**9 + 4333523094476748288*x0**24*x1**8 + 1359741493569873408*x0**25*x1**7 + 2868127923362612736*x0**26*x1**6 + 1117363258488942080*x0**27*x1**5 + 6496634226022710784*x0**28*x1**4 + 103818320911271424*x0**29*x1**3 + 488658578104600064*x0**30*x1**2 + 18251159610004999680*x0**31*x1 + 10412781181975305728*x0**32 + 16548171920898597888*x1**33 + 5583109433595892736*x0*x1**32 + 3675960375135415808*x0**2*x1**31 + 12473046247849759232*x0**3*x1**30 + 7028520935507832320*x0**4*x1**29 + 16930797547690676736*x0**5*x1**28 + 11512684994249678336*x0**6*x1**27 + 16491797892868956672*x0**7*x1**26 + 15025884357806468608*x0**8*x1**25 + 14248923848567239168*x0**9*x1**24 + 1330764194969536000*x0**10*x1**23 + 15336797538460858880*x0**11*x1**22 + 16034241301613741568*x0**12*x1**21 + 785412745473980928*x0**13*x1**20 + 6937341224349488640*x0**14*x1**19 + 2849997286201158144*x0**15*x1**18 + 1135006177438895616*x0**16*x1**17 + 3482143410143298048*x0**17*x1**16 + 2791363636894157312*x0**18*x1**15 + 9652379474292241920*x0**19*x1**14 + 7550666279519635968*x0**20*x1**13 + 11017248540218996224*x0**21*x1**12 + 3324150440113969664*x0**22*x1**11 + 14326887154295186944*x0**23*x1**10 + 14351213706880330240*x0**24*x1**9 + 16475056202102552064*x0**25*x1**8 + 3464053238765142528*x0**26*x1**7 + 5224859802986380800*x0**27*x1**6 + 13362340671065883136*x0**28*x1**5 + 14882092529080466944*x0**29*x1**4 + 7842370692330926592*x0**30*x1**3 + 11930709613448924672*x0**31*x1**2 + 1544364914155320832*x0**32*x1 + 17108072015547105792*x0**33 + 8244176583627937792*x1**34 + 16192160759464977920*x0*x1**33 + 6156135400048877056*x0**2*x1**32 + 3062245037603708928*x0**3*x1**31 + 9668397629510361088*x0**4*x1**30 + 11658303092227051520*x0**5*x1**29 + 16163365193925611520*x0**6*x1**28 + 12239597185420271616*x0**7*x1**27 + 13656056005237030912*x0**8*x1**26 + 1390112356204701696*x0**9*x1**25 + 4950922363304904704*x0**10*x1**24 + 707649064807522304*x0**11*x1**23 + 16943526289135247360*x0**12*x1**22 + 2211564052813836288*x0**13*x1**21 + 7313404095491125248*x0**14*x1**20 + 2659940560319193088*x0**15*x1**19 + 6747759497776410624*x0**16*x1**18 + 6147651381937251328*x0**17*x1**17 + 14025228865117025280*x0**18*x1**16 + 4642479618567184384*x0**19*x1**15 + 9697164398327422976*x0**20*x1**14 + 10670279375220600832*x0**21*x1**13 + 9328056342520016896*x0**22*x1**12 + 9392170139951489024*x0**23*x1**11 + 16171063214913257472*x0**24*x1**10 + 8001582914296932352*x0**25*x1**9 + 7900733926219843584*x0**26*x1**8 + 13286534768684441600*x0**27*x1**7 + 12227290643817127936*x0**28*x1**6 + 7563015385755389952*x0**29*x1**5 + 16944283394318585856*x0**30*x1**4 + 5633213572130922496*x0**31*x1**3 + 18048952142430697472*x0**32*x1**2 + 1543204859494180352*x0**33*x1 + 15048908507725929984*x0**34 + 2110639693301173248*x1**35 + 4766311470316079104*x0*x1**34 + 11767901850730069504*x0**2*x1**33 + 15686551805559669248*x0**3*x1**32 + 11299048203336744960*x0**4*x1**31 + 14978311286773145600*x0**5*x1**30 + 10676755579394494464*x0**6*x1**29 + 6569833868275134464*x0**7*x1**28 + 5084662815042891776*x0**8*x1**27 + 8753245059926278144*x0**9*x1**26 + 14126397067300489216*x0**10*x1**25 + 11646117142890674176*x0**11*x1**24 + 10174966245569183744*x0**12*x1**23 + 17842687568957399040*x0**13*x1**22 + 1241773435141222400*x0**14*x1**21 + 10315335982863831040*x0**15*x1**20 + 12541644718760880128*x0**16*x1**19 + 6130738471184306176*x0**17*x1**18 + 9470979805600445440*x0**18*x1**17 + 9857884610359872512*x0**19*x1**16 + 1884993891868377088*x0**20*x1**15 + 13203554535018930176*x0**21*x1**14 + 2920203398549204992*x0**22*x1**13 + 5798076964743925760*x0**23*x1**12 + 1609812140804222976*x0**24*x1**11 + 15686419257519964160*x0**25*x1**10 + 15727053255104055296*x0**26*x1**9 + 9906720990845564928*x0**27*x1**8 + 15986380733485465600*x0**28*x1**7 + 13295668829491077120*x0**29*x1**6 + 13265275929442160640*x0**30*x1**5 + 14114409811784839168*x0**31*x1**4 + 16182890949708229632*x0**32*x1**3 + 5393392637465421824*x0**33*x1**2 + 12835994569019063808*x0**34*x1 + 3408858061891813888*x0**35 + 5117173533067399168*x1**36 + 3141593031418622464*x0*x1**35 + 13503668269838746112*x0**2*x1**34 + 5623266469916790272*x0**3*x1**33 + 11898959286616649216*x0**4*x1**32 + 9126673178683183104*x0**5*x1**31 + 1771793256690978816*x0**6*x1**30 + 373788346839642112*x0**7*x1**29 + 9949036198413299712*x0**8*x1**28 + 9679670110884157440*x0**9*x1**27 + 2179200581007054848*x0**10*x1**26 + 9375784137452775424*x0**11*x1**25 + 170824050905475072*x0**12*x1**24 + 7222693970015399936*x0**13*x1**23 + 622961357037056000*x0**14*x1**22 + 5772210595533754368*x0**15*x1**21 + 17484612823455428608*x0**16*x1**20 + 9472253799358368768*x0**17*x1**19 + 12352474803959163904*x0**18*x1**18 + 5604534041792361472*x0**19*x1**17 + 13123935560650355712*x0**20*x1**16 + 10277070744925237248*x0**21*x1**15 + 1701759823118036992*x0**22*x1**14 + 1811525227507961856*x0**23*x1**13 + 14660767673665458176*x0**24*x1**12 + 5896433307091998720*x0**25*x1**11 + 9659312900119386112*x0**26*x1**10 + 11064268667371792384*x0**27*x1**9 + 9416280189520455680*x0**28*x1**8 + 6797676556275961856*x0**29*x1**7 + 4326960644563062784*x0**30*x1**6 + 7473069504668102656*x0**31*x1**5 + 12338564247849484288*x0**32*x1**4 + 14529078755493368320*x0**33*x1**3 + 12641090386041987584*x0**34*x1**2 + 12887835881996283392*x0**35*x1 + 1656794814415834624*x0**36 + 10568101363740369920*x1**37 + 8210581810417363968*x0*x1**36 + 14121589072722603520*x0**2*x1**35 + 11368831703836305920*x0**3*x1**34 + 7530860570737083904*x0**4*x1**33 + 3913186312472723968*x0**5*x1**32 + 14963444642662445056*x0**6*x1**31 + 1355591826902077440*x0**7*x1**30 + 18352013864335613952*x0**8*x1**29 + 4870009806196400128*x0**9*x1**28 + 6267224257561896960*x0**10*x1**27 + 2023299542287185920*x0**11*x1**26 + 711601398973032448*x0**12*x1**25 + 12479266794755880960*x0**13*x1**24 + 14375777115428704256*x0**14*x1**23 + 14305916726973820928*x0**15*x1**22 + 9436655697776994304*x0**16*x1**21 + 13489084938213177344*x0**17*x1**20 + 6253722475004867584*x0**18*x1**19 + 12796587001860054016*x0**19*x1**18 + 9022597432809792512*x0**20*x1**17 + 17475550536989903872*x0**21*x1**16 + 10432694679662956544*x0**22*x1**15 + 6323270861117403136*x0**23*x1**14 + 9390334188991037440*x0**24*x1**13 + 3269752001015267328*x0**25*x1**12 + 14281309655961528320*x0**26*x1**11 + 12134120439369873408*x0**27*x1**10 + 13728223389093160960*x0**28*x1**9 + 7560096256467826688*x0**29*x1**8 + 14591694847888764928*x0**30*x1**7 + 1142019468598972416*x0**31*x1**6 + 15686166081742525440*x0**32*x1**5 + 1933353131487640576*x0**33*x1**4 + 8277201651249417728*x0**34*x1**3 + 11293648970175793664*x0**35*x1**2 + 1991859142764922368*x0**36*x1 + 17145415252550943232*x0**37 + 8920387658527795200*x1**38 + 6540655142760203776*x0*x1**37 + 13687708409264418304*x0**2*x1**36 + 5609450932021394432*x0**3*x1**35 + 1461759479641244672*x0**4*x1**34 + 16088778576689139200*x0**5*x1**33 + 6946644855080793600*x0**6*x1**32 + 1882562117870166016*x0**7*x1**31 + 3681565211727970304*x0**8*x1**30 + 4293893422827010048*x0**9*x1**29 + 2858418754695436288*x0**10*x1**28 + 5177208472681975808*x0**11*x1**27 + 2312622379222642688*x0**12*x1**26 + 4896055591771027456*x0**13*x1**25 + 13102932679306270720*x0**14*x1**24 + 4668962810087071744*x0**15*x1**23 + 16168750796272013312*x0**16*x1**22 + 11856616866103841792*x0**17*x1**21 + 558004051271222272*x0**18*x1**20 + 836336535749879808*x0**19*x1**19 + 14328070643386046464*x0**20*x1**18 + 15594408420135855104*x0**21*x1**17 + 4878726067346641920*x0**22*x1**16 + 3294041508106002432*x0**23*x1**15 + 13317913213282320384*x0**24*x1**14 + 14431113764571928576*x0**25*x1**13 + 1912096885353609216*x0**26*x1**12 + 8731743833907335168*x0**27*x1**11 + 17254298603430195200*x0**28*x1**10 + 4016066486587578368*x0**29*x1**9 + 7235113453348681728*x0**30*x1**8 + 12558742060242509824*x0**31*x1**7 + 4247694448421160960*x0**32*x1**6 + 5321312947006543360*x0**33*x1**5 + 13449808381531209216*x0**34*x1**4 + 396885933743864832*x0**35*x1**3 + 4190402921753953280*x0**36*x1**2 + 12020531588020928000*x0**37*x1 + 6149192191730181632*x0**38 + 16215530120667933696*x1**39 + 2350024119884288000*x0*x1**38 + 11103580637391582720*x0**2*x1**37 + 18385949593284600320*x0**3*x1**36 + 5893470229019502592*x0**4*x1**35 + 4520019174473012224*x0**5*x1**34 + 15901048589585049088*x0**6*x1**33 + 11891493592000965120*x0**7*x1**32 + 12655869264034140160*x0**8*x1**31 + 8385051542662553600*x0**9*x1**30 + 7176515564041533440*x0**10*x1**29 + 9816143330574018560*x0**11*x1**28 + 9418851023270723584*x0**12*x1**27 + 3310571315728297984*x0**13*x1**26 + 3131523559781685248*x0**14*x1**25 + 16579756833393518592*x0**15*x1**24 + 6033971960034805760*x0**16*x1**23 + 2475485704239644672*x0**17*x1**22 + 11915405939877032960*x0**18*x1**21 + 13359423621208497152*x0**19*x1**20 + 17602138524704280576*x0**20*x1**19 + 8878319809850013696*x0**21*x1**18 + 4161872042599478272*x0**22*x1**17 + 7637962979001510912*x0**23*x1**16 + 3356821404798259200*x0**24*x1**15 + 4034168740662370304*x0**25*x1**14 + 9765509120129456128*x0**26*x1**13 + 13555592324771661824*x0**27*x1**12 + 711679801929269248*x0**28*x1**11 + 11363000826711977984*x0**29*x1**10 + 15220950000626595840*x0**30*x1**9 + 3749492360886257664*x0**31*x1**8 + 2712958908687739904*x0**32*x1**7 + 15976260391155331072*x0**33*x1**6 + 17281626997792048640*x0**34*x1**5 + 11644206550287798784*x0**35*x1**4 + 14103670003776194560*x0**36*x1**3 + 6823140950970340352*x0**37*x1**2 + 396512517057598976*x0**38*x1 + 10023442448058251776*x0**39 + 16935463419326398464*x1**40 + 9263320784850043392*x0*x1**39 + 14659348672819462656*x0**2*x1**38 + 5804441616745087488*x0**3*x1**37 + 10157453850978529792*x0**4*x1**36 + 3322771627036730880*x0**5*x1**35 + 14369091179766203904*x0**6*x1**34 + 11820797761075934720*x0**7*x1**33 + 9143458380702803456*x0**8*x1**32 + 717401955814262784*x0**9*x1**31 + 13399455208953137152*x0**10*x1**30 + 14547948280157128704*x0**11*x1**29 + 12251693231692974080*x0**12*x1**28 + 11052026600840345600*x0**13*x1**27 + 8792432684293044224*x0**14*x1**26 + 17357996709166741504*x0**15*x1**25 + 11911639893777922048*x0**16*x1**24 + 10096759560705899520*x0**17*x1**23 + 7727741220859083776*x0**18*x1**22 + 6935925331923106816*x0**19*x1**21 + 15304276927994797056*x0**20*x1**20 + 5685466335722214400*x0**21*x1**19 + 10415563485127005184*x0**22*x1**18 + 935973778147800064*x0**23*x1**17 + 6678485942143325184*x0**24*x1**16 + 7200953005612148736*x0**25*x1**15 + 2100779467630651392*x0**26*x1**14 + 9358917718675335168*x0**27*x1**13 + 12307092629842061312*x0**28*x1**12 + 9611719630587467776*x0**29*x1**11 + 12240879940677212160*x0**30*x1**10 + 4454557041382742016*x0**31*x1**9 + 909294593425184768*x0**32*x1**8 + 4638947817487716864*x0**33*x1**7 + 7250191537049145856*x0**34*x1**6 + 8345777909269090816*x0**35*x1**5 + 5215774519025320448*x0**36*x1**4 + 9394950550659282432*x0**37*x1**3 + 3849907880202507776*x0**38*x1**2 + 1940444292301833728*x0**39*x1 + 12207305863002797568*x0**40 + 10448557437303739392*x1**41 + 7904637828256117760*x0*x1**40 + 16726220194358784*x0**2*x1**39 + 8696966636089027072*x0**3*x1**38 + 15525169216114288128*x0**4*x1**37 + 8010146898834266624*x0**5*x1**36 + 1751366093908099584*x0**6*x1**35 + 14393588367618485760*x0**7*x1**34 + 10107198439740214784*x0**8*x1**33 + 15383446435733883392*x0**9*x1**32 + 1267107533770348544*x0**10*x1**31 + 12511910757598816256*x0**11*x1**30 + 15709211557830957056*x0**12*x1**29 + 7475004639515895808*x0**13*x1**28 + 550709451984463872*x0**14*x1**27 + 293276167582701568*x0**15*x1**26 + 7345764922526593024*x0**16*x1**25 + 9673646959576834048*x0**17*x1**24 + 8793775656542745600*x0**18*x1**23 + 12123200567465487360*x0**19*x1**22 + 13876473070112928768*x0**20*x1**21 + 11499319427493694464*x0**21*x1**20 + 14653253259239988224*x0**22*x1**19 + 14201757256290173952*x0**23*x1**18 + 13370532204003488768*x0**24*x1**17 + 14318573399242849280*x0**25*x1**16 + 3176022442237786112*x0**26*x1**15 + 3315492614166923264*x0**27*x1**14 + 17110272616811870208*x0**28*x1**13 + 5008546727864322048*x0**29*x1**12 + 17154271096352966656*x0**30*x1**11 + 16961338552589957120*x0**31*x1**10 + 10272832929705618432*x0**32*x1**9 + 16739732588935560192*x0**33*x1**8 + 16482045096451855872*x0**34*x1**7 + 13352595743622330880*x0**35*x1**6 + 932905035901228544*x0**36*x1**5 + 14639769632315714048*x0**37*x1**4 + 6264528025285014016*x0**38*x1**3 + 6689334854435421696*x0**39*x1**2 + 5502561160055246336*x0**40*x1 + 14012485558264441344*x0**41 + 7343036819913365504*x1**42 + 16220633157617937920*x0*x1**41 + 7499011847629923840*x0**2*x1**40 + 16197177450722613248*x0**3*x1**39 + 2077460840306747392*x0**4*x1**38 + 15211489308966515712*x0**5*x1**37 + 14177936590370161664*x0**6*x1**36 + 586768303070001152*x0**7*x1**35 + 3732886618104885248*x0**8*x1**34 + 17178163902330701312*x0**9*x1**33 + 3194744599316504064*x0**10*x1**32 + 5930744215988854784*x0**11*x1**31 + 5270513216571023360*x0**12*x1**30 + 12702632877138112512*x0**13*x1**29 + 5759177360000790528*x0**14*x1**28 + 8769396099235176448*x0**15*x1**27 + 10250090879787077632*x0**16*x1**26 + 13398611344182311936*x0**17*x1**25 + 647578279804857344*x0**18*x1**24 + 9542834974005579776*x0**19*x1**23 + 10031863080163717120*x0**20*x1**22 + 14268054944793016320*x0**21*x1**21 + 15619024036463613952*x0**22*x1**20 + 17660436556954243072*x0**23*x1**19 + 973231976293810176*x0**24*x1**18 + 4292278338889997312*x0**25*x1**17 + 11072691415688137728*x0**26*x1**16 + 4958424159041445888*x0**27*x1**15 + 18390737758288945152*x0**28*x1**14 + 11956967602450264064*x0**29*x1**13 + 8430371453010751488*x0**30*x1**12 + 13446458607474597888*x0**31*x1**11 + 8590078010995202048*x0**32*x1**10 + 9285365644551099904*x0**33*x1**9 + 14456591961653521920*x0**34*x1**8 + 6661653594078844928*x0**35*x1**7 + 14810755556739049472*x0**36*x1**6 + 12229774956346635264*x0**37*x1**5 + 15702001966020510720*x0**38*x1**4 + 68212959237175296*x0**39*x1**3 + 8715173141326008320*x0**40*x1**2 + 3899275907981843968*x0**41*x1 + 9273653422898812416*x0**42 + 4671623204680940544*x1**43 + 15746756291725416448*x0*x1**42 + 8019225037908669952*x0**2*x1**41 + 10349584617495845376*x0**3*x1**40 + 3621270931343351808*x0**4*x1**39 + 3228183578404730880*x0**5*x1**38 + 8470966920409232384*x0**6*x1**37 + 4376286367672220672*x0**7*x1**36 + 9025366975412814848*x0**8*x1**35 + 11875730458955038720*x0**9*x1**34 + 11948176643190275584*x0**10*x1**33 + 15175417557399773696*x0**11*x1**32 + 15938935523463741440*x0**12*x1**31 + 3963999579217649664*x0**13*x1**30 + 16788581336711393280*x0**14*x1**29 + 7905670722773864448*x0**15*x1**28 + 197116512501307392*x0**16*x1**27 + 3955455763152850944*x0**17*x1**26 + 14345765694447832064*x0**18*x1**25 + 8335041758481716224*x0**19*x1**24 + 717267464756494336*x0**20*x1**23 + 2431289386400452608*x0**21*x1**22 + 2551378154624366592*x0**22*x1**21 + 10841545397614503936*x0**23*x1**20 + 8568492295651379200*x0**24*x1**19 + 16935786029692854272*x0**25*x1**18 + 10141063335048006656*x0**26*x1**17 + 9468673464113280000*x0**27*x1**16 + 7412144505893339136*x0**28*x1**15 + 2133661689330491392*x0**29*x1**14 + 10732639245812592640*x0**30*x1**13 + 2750643240973717504*x0**31*x1**12 + 8309018806126930944*x0**32*x1**11 + 12066820971708520448*x0**33*x1**10 + 4721556908350029312*x0**34*x1**9 + 15415077757777220096*x0**35*x1**8 + 8152915864884043776*x0**36*x1**7 + 6638912993667315712*x0**37*x1**6 + 8513946711403639808*x0**38*x1**5 + 6244200694173940736*x0**39*x1**4 + 11763483282803602432*x0**40*x1**3 + 7160921703350624256*x0**41*x1**2 + 18080641251203479040*x0**42*x1 + 15691916582545435136*x0**43 + 16235998746113921024*x1**44 + 5700291242521516544*x0*x1**43 + 2538279310589913600*x0**2*x1**42 + 15871259221988355584*x0**3*x1**41 + 7011601958768365056*x0**4*x1**40 + 11935225741777466368*x0**5*x1**39 + 14625931842489160704*x0**6*x1**38 + 12557567981922298880*x0**7*x1**37 + 8834072731818327040*x0**8*x1**36 + 643126667592792576*x0**9*x1**35 + 6997759060106152448*x0**10*x1**34 + 10651988949981593088*x0**11*x1**33 + 3048693606375554560*x0**12*x1**32 + 12493934727708782592*x0**13*x1**31 + 6134910061089206272*x0**14*x1**30 + 7485854821828415488*x0**15*x1**29 + 1869766639722295296*x0**16*x1**28 + 12300703148613694464*x0**17*x1**27 + 3968800832791581696*x0**18*x1**26 + 788586577293073408*x0**19*x1**25 + 8565489504200858624*x0**20*x1**24 + 12937189285098715136*x0**21*x1**23 + 379419249802229760*x0**22*x1**22 + 10438771789068879872*x0**23*x1**21 + 7583457854861076480*x0**24*x1**20 + 2701197720768283648*x0**25*x1**19 + 13567949876052861952*x0**26*x1**18 + 8452108249859058688*x0**27*x1**17 + 18235284513310895104*x0**28*x1**16 + 7459332132858507264*x0**29*x1**15 + 15841940145117704192*x0**30*x1**14 + 13026884463596851200*x0**31*x1**13 + 7957519286191415296*x0**32*x1**12 + 6184169607008940544*x0**33*x1**11 + 12917757578229596672*x0**34*x1**10 + 18329008452140054016*x0**35*x1**9 + 15622822494214346240*x0**36*x1**8 + 11434488515280843776*x0**37*x1**7 + 9898502950311482368*x0**38*x1**6 + 2858873168200002560*x0**39*x1**5 + 12348319604466121728*x0**40*x1**4 + 80613289969140224*x0**41*x1**3 + 1468313030725571072*x0**42*x1**2 + 16547382024156337664*x0**43*x1 + 5225966901884966400*x0**44 + 4139910357228411904*x1**45 + 9808404981490476032*x0*x1**44 + 10982433316668753408*x0**2*x1**43 + 6823957520009609728*x0**3*x1**42 + 13944550688572163584*x0**4*x1**41 + 4785230622582678016*x0**5*x1**40 + 8141273591754771456*x0**6*x1**39 + 9473679219659932672*x0**7*x1**38 + 14154001243283810304*x0**8*x1**37 + 13339945720087750656*x0**9*x1**36 + 16698605450022151680*x0**10*x1**35 + 15447572210876015104*x0**11*x1**34 + 11174909679145672192*x0**12*x1**33 + 8369683501011700224*x0**13*x1**32 + 14093003737218240512*x0**14*x1**31 + 5915895905380036608*x0**15*x1**30 + 13799873930744498176*x0**16*x1**29 + 1648066513679890432*x0**17*x1**28 + 7963289984127618048*x0**18*x1**27 + 3206736341157266432*x0**19*x1**26 + 6157477571181407232*x0**20*x1**25 + 8619131925166810112*x0**21*x1**24 + 4814995671223306240*x0**22*x1**23 + 10738333806947719168*x0**23*x1**22 + 1387290776187576320*x0**24*x1**21 + 13184389432083628032*x0**25*x1**20 + 6127680958925683712*x0**26*x1**19 + 13987624428000318464*x0**27*x1**18 + 8902675557458938880*x0**28*x1**17 + 14324581670332695552*x0**29*x1**16 + 2139751378273226752*x0**30*x1**15 + 13762483785620779008*x0**31*x1**14 + 9699997716482243584*x0**32*x1**13 + 11018146390529688576*x0**33*x1**12 + 13429236750556868096*x0**34*x1**11 + 5668611533760543232*x0**35*x1**10 + 6584352865889603072*x0**36*x1**9 + 2874736042518772224*x0**37*x1**8 + 11684821574617598976*x0**38*x1**7 + 14364149119195431936*x0**39*x1**6 + 3979961556863997952*x0**40*x1**5 + 4890794149803735040*x0**41*x1**4 + 3883240391908728320*x0**42*x1**3 + 17939301555374606848*x0**43*x1**2 + 1894896452901246464*x0**44*x1 + 4033518899675984384*x0**45 + 11980087992657410048*x1**46 + 11097736319171074560*x0*x1**45 + 12232765200868397568*x0**2*x1**44 + 9366897326261152768*x0**3*x1**43 + 3162931319931638784*x0**4*x1**42 + 9821201288780475904*x0**5*x1**41 + 3952796527650944512*x0**6*x1**40 + 17949401766587983872*x0**7*x1**39 + 17559887781053485056*x0**8*x1**38 + 3783689072315735552*x0**9*x1**37 + 9387678817264600576*x0**10*x1**36 + 3724918394178690048*x0**11*x1**35 + 11031813670397056000*x0**12*x1**34 + 5015595564440066560*x0**13*x1**33 + 8862748670784614912*x0**14*x1**32 + 1328678773749178368*x0**15*x1**31 + 13864092357501128704*x0**16*x1**30 + 6597330736276759552*x0**17*x1**29 + 11182667032882222080*x0**18*x1**28 + 16790988173215045632*x0**19*x1**27 + 1911335808744368128*x0**20*x1**26 + 17797535207654222848*x0**21*x1**25 + 8526352950801873920*x0**22*x1**24 + 7549164980544827392*x0**23*x1**23 + 2964719332090560512*x0**24*x1**22 + 15442794470402366464*x0**25*x1**21 + 14636494258695793664*x0**26*x1**20 + 4335444397024204800*x0**27*x1**19 + 8631900152702552064*x0**28*x1**18 + 11220737093670917120*x0**29*x1**17 + 12165266229708162048*x0**30*x1**16 + 6763505698496438272*x0**31*x1**15 + 2803366296154322944*x0**32*x1**14 + 7933054587998282240*x0**33*x1**13 + 4542358067531004416*x0**34*x1**12 + 6945908137167436800*x0**35*x1**11 + 226491814026724352*x0**36*x1**10 + 11388365499634877952*x0**37*x1**9 + 13407866801562349056*x0**38*x1**8 + 10158472466173126656*x0**39*x1**7 + 14265182059463356416*x0**40*x1**6 + 9582552735475672576*x0**41*x1**5 + 8130843236275056128*x0**42*x1**4 + 9247295952257043456*x0**43*x1**3 + 1854154034572174336*x0**44*x1**2 + 3315307018312495616*x0**45*x1 + 9872775246468278784*x0**46 + 1559732035608237056*x1**47 + 6439268841888718848*x0*x1**46 + 666566871384603136*x0**2*x1**45 + 8068040824533179904*x0**3*x1**44 + 1234260496148715520*x0**4*x1**43 + 15505394743870256128*x0**5*x1**42 + 13762844544711005696*x0**6*x1**41 + 6982373167825127936*x0**7*x1**40 + 11411649947722327040*x0**8*x1**39 + 7606052300115687424*x0**9*x1**38 + 3067985570472062464*x0**10*x1**37 + 15196746820201490944*x0**11*x1**36 + 10273348191690448896*x0**12*x1**35 + 17414383576610489344*x0**13*x1**34 + 12583997017245098496*x0**14*x1**33 + 5118047105367764480*x0**15*x1**32 + 10747933437922600960*x0**16*x1**31 + 16215963408023773184*x0**17*x1**30 + 1379288734902936576*x0**18*x1**29 + 13718496584230685696*x0**19*x1**28 + 14941231053386158080*x0**20*x1**27 + 9781722978048452608*x0**21*x1**26 + 4553954740497202176*x0**22*x1**25 + 8988647533954581504*x0**23*x1**24 + 2091518392074250240*x0**24*x1**23 + 7464421374073622528*x0**25*x1**22 + 12216415247494970368*x0**26*x1**21 + 9504388588119028736*x0**27*x1**20 + 9362860127230271488*x0**28*x1**19 + 4124470949532559360*x0**29*x1**18 + 13517768320691184640*x0**30*x1**17 + 16458053549768326144*x0**31*x1**16 + 13863661541399184384*x0**32*x1**15 + 16822031597427245056*x0**33*x1**14 + 13166043333188452864*x0**34*x1**13 + 14101169991689975296*x0**35*x1**12 + 13600474933265815552*x0**36*x1**11 + 7359625838599726080*x0**37*x1**10 + 10999213727025303040*x0**38*x1**9 + 16210544128616092160*x0**39*x1**8 + 8097425303251104768*x0**40*x1**7 + 387133585776949248*x0**41*x1**6 + 17935221111078188544*x0**42*x1**5 + 9203673090426798592*x0**43*x1**4 + 6765371105063157760*x0**44*x1**3 + 4609066889415283712*x0**45*x1**2 + 9409844244625742336*x0**46*x1 + 16822380707382310400*x0**47 + 17780312310650707968*x1**48 + 15806637951010258432*x0*x1**47 + 17907609654208503296*x0**2*x1**46 + 11294670109254161920*x0**3*x1**45 + 14053700221057345024*x0**4*x1**44 + 7225029139957159424*x0**5*x1**43 + 15322626876628522496*x0**6*x1**42 + 2328981942245453312*x0**7*x1**41 + 15327186674747629056*x0**8*x1**40 + 3764506480539660800*x0**9*x1**39 + 4336492636411384320*x0**10*x1**38 + 15972116221223606784*x0**11*x1**37 + 5203258973924542976*x0**12*x1**36 + 2556145682155743744*x0**13*x1**35 + 5128427984888867328*x0**14*x1**34 + 10599358715897516544*x0**15*x1**33 + 10262583649640612352*x0**16*x1**32 + 11257586523008000*x0**17*x1**31 + 16031101807562599424*x0**18*x1**30 + 2542291602481208320*x0**19*x1**29 + 10562222168337972224*x0**20*x1**28 + 18429457375968173056*x0**21*x1**27 + 5738541146035297280*x0**22*x1**26 + 5483135957489781760*x0**23*x1**25 + 7663383190191139840*x0**24*x1**24 + 7422536288511763456*x0**25*x1**23 + 17339970550337070080*x0**26*x1**22 + 2425596550051394560*x0**27*x1**21 + 8871636587041739776*x0**28*x1**20 + 14041496740186018816*x0**29*x1**19 + 6004630328131834880*x0**30*x1**18 + 7130312608639329280*x0**31*x1**17 + 18093689820989268992*x0**32*x1**16 + 3025227475659453952*x0**33*x1**15 + 3484754991228835328*x0**34*x1**14 + 783199425628515840*x0**35*x1**13 + 13128881036619054592*x0**36*x1**12 + 4052904120424721920*x0**37*x1**11 + 17778032383523663360*x0**38*x1**10 + 4519880364109014528*x0**39*x1**9 + 6456304135815616000*x0**40*x1**8 + 13304117473987480064*x0**41*x1**7 + 4196891647747795456*x0**42*x1**6 + 2381576642446807552*x0**43*x1**5 + 17232673454425195008*x0**44*x1**4 + 18120069257187058176*x0**45*x1**3 + 17407752436928864768*x0**46*x1**2 + 2527966146928945664*x0**47*x1 + 11116271633689483776*x0**48 + 11678680663103319040*x1**49 + 12587357456861633536*x0*x1**48 + 13632098467195128320*x0**2*x1**47 + 18303836485818586624*x0**3*x1**46 + 12643210492127955456*x0**4*x1**45 + 15178010726852523520*x0**5*x1**44 + 8265631894354665984*x0**6*x1**43 + 15058949801122692608*x0**7*x1**42 + 8444216693101226496*x0**8*x1**41 + 1770808385595393536*x0**9*x1**40 + 4071441679747343872*x0**10*x1**39 + 11826036901525223936*x0**11*x1**38 + 13913132928677545472*x0**12*x1**37 + 5795068455228078592*x0**13*x1**36 + 9900202047033343488*x0**14*x1**35 + 10170295946499318272*x0**15*x1**34 + 16077292432990855680*x0**16*x1**33 + 14547601904093607424*x0**17*x1**32 + 6664334472040866816*x0**18*x1**31 + 1120974726329031680*x0**19*x1**30 + 11105387835471750144*x0**20*x1**29 + 10400315925639502848*x0**21*x1**28 + 5399116794865030144*x0**22*x1**27 + 4334876958720371712*x0**23*x1**26 + 12990269768017298432*x0**24*x1**25 + 7401830477274242048*x0**25*x1**24 + 8892854934755642368*x0**26*x1**23 + 434735657546363904*x0**27*x1**22 + 11349653862569561088*x0**28*x1**21 + 7224798235134571520*x0**29*x1**20 + 6252281755288599552*x0**30*x1**19 + 954595408086483968*x0**31*x1**18 + 9028945867389251584*x0**32*x1**17 + 5389285800856733696*x0**33*x1**16 + 6543291447281086976*x0**34*x1**15 + 14465839570704084480*x0**35*x1**14 + 9809924905483776512*x0**36*x1**13 + 173690590279110144*x0**37*x1**12 + 12438245286145534464*x0**38*x1**11 + 12289953615194781184*x0**39*x1**10 + 1853570681849459200*x0**40*x1**9 + 11948795316047347200*x0**41*x1**8 + 7687162428855594496*x0**42*x1**7 + 5375616241976586752*x0**43*x1**6 + 18282017126311972352*x0**44*x1**5 + 11069497192839033344*x0**45*x1**4 + 3068067100391319040*x0**46*x1**3 + 9778636639476759040*x0**47*x1**2 + 5096868983691764224*x0**48*x1 + 9039748055328278016*x0**49 + 40526322154539008*x1**50 + 9807408472960422400*x0*x1**49 + 6371361877452126720*x0**2*x1**48 + 1973448814682091520*x0**3*x1**47 + 14528377846700527616*x0**4*x1**46 + 2384649401570592768*x0**5*x1**45 + 6070877667896338432*x0**6*x1**44 + 8080040250007949312*x0**7*x1**43 + 14245177904074170368*x0**8*x1**42 + 3375962891457643520*x0**9*x1**41 + 9089426642661413888*x0**10*x1**40 + 4386253854938173440*x0**11*x1**39 + 12934473186082304000*x0**12*x1**38 + 650174596478707712*x0**13*x1**37 + 15239789361774577664*x0**14*x1**36 + 13164140767851982848*x0**15*x1**35 + 13976889826840750080*x0**16*x1**34 + 10126978089813765632*x0**17*x1**33 + 3362507985867326976*x0**18*x1**32 + 10226465727096414208*x0**19*x1**31 + 12308454942584283136*x0**20*x1**30 + 15620488482272579584*x0**21*x1**29 + 2634744401273507840*x0**22*x1**28 + 2392036687020269568*x0**23*x1**27 + 9434497832413659136*x0**24*x1**26 + 4177762818767468544*x0**25*x1**25 + 15170846977612617728*x0**26*x1**24 + 10693944004809138176*x0**27*x1**23 + 13947905681267122176*x0**28*x1**22 + 6589891891965104128*x0**29*x1**21 + 7453543995841826816*x0**30*x1**20 + 3797017898182598656*x0**31*x1**19 + 11984146183326455808*x0**32*x1**18 + 299958926612123136*x0**33*x1**17 + 2016331643468916224*x0**34*x1**16 + 16613288053527441408*x0**35*x1**15 + 16251699645513371648*x0**36*x1**14 + 312091420327897088*x0**37*x1**13 + 4555542405872519168*x0**38*x1**12 + 13804812546189316096*x0**39*x1**11 + 16803266212366573568*x0**40*x1**10 + 17124420541542843392*x0**41*x1**9 + 12352763847012684800*x0**42*x1**8 + 651959101098872832*x0**43*x1**7 + 13356186110802292736*x0**44*x1**6 + 1290321207958882304*x0**45*x1**5 + 13284688265610422272*x0**46*x1**4 + 647087641746960384*x0**47*x1**3 + 1453308999319353344*x0**48*x1**2 + 480217982638076416*x0**49*x1 + 2037206674275392000*x0**50 + 18329296043979121664*x1**51 + 3571696797289986048*x0*x1**50 + 5724778542614002176*x0**2*x1**49 + 17398115423830353408*x0**3*x1**48 + 7084353335373463552*x0**4*x1**47 + 10496932240074813440*x0**5*x1**46 + 11597952965931911168*x0**6*x1**45 + 8698800933087664128*x0**7*x1**44 + 18397961657775642624*x0**8*x1**43 + 10621979903199895552*x0**9*x1**42 + 5352685471130383360*x0**10*x1**41 + 11142473325087730688*x0**11*x1**40 + 6495962644054089728*x0**12*x1**39 + 1910004895858774016*x0**13*x1**38 + 1220045008821102592*x0**14*x1**37 + 8631019857811224576*x0**15*x1**36 + 9967069878177352704*x0**16*x1**35 + 14230508503764932608*x0**17*x1**34 + 11332091008173155840*x0**18*x1**33 + 3476931996123863552*x0**19*x1**32 + 11436022235411546112*x0**20*x1**31 + 3704103628809035776*x0**21*x1**30 + 7904731463087329280*x0**22*x1**29 + 1874637722410725376*x0**23*x1**28 + 16377507554956382208*x0**24*x1**27 + 12991251719263256576*x0**25*x1**26 + 12986753485179017216*x0**26*x1**25 + 5087782023047669760*x0**27*x1**24 + 10622864989079453696*x0**28*x1**23 + 16113977383540400128*x0**29*x1**22 + 14133007328838848512*x0**30*x1**21 + 7381962676268576768*x0**31*x1**20 + 5748426788960467968*x0**32*x1**19 + 15055017830990505984*x0**33*x1**18 + 6501521128651433472*x0**34*x1**17 + 3073239275341462016*x0**35*x1**16 + 15386549167980314624*x0**36*x1**15 + 14233069247855456256*x0**37*x1**14 + 13399276815335565312*x0**38*x1**13 + 15887675340155025408*x0**39*x1**12 + 3425931582170503168*x0**40*x1**11 + 12447057455489622016*x0**41*x1**10 + 17404673961751933952*x0**42*x1**9 + 12809925419636370432*x0**43*x1**8 + 13257904581298102272*x0**44*x1**7 + 5900740932845506560*x0**45*x1**6 + 16302669622909093888*x0**46*x1**5 + 5838764126207842304*x0**47*x1**4 + 15143831916969694208*x0**48*x1**3 + 11791510336604555264*x0**49*x1**2 + 5634862757384894976*x0**50*x1 + 9906064793980454400*x0**51 + 3838818677378691072*x1**52 + 17269099348960048640*x0*x1**51 + 3880100311796835840*x0**2*x1**50 + 11020064043073206784*x0**3*x1**49 + 9364065466833197568*x0**4*x1**48 + 17978606366753196032*x0**5*x1**47 + 1757706340491446272*x0**6*x1**46 + 10109158756690892800*x0**7*x1**45 + 3229495661992114176*x0**8*x1**44 + 3832624931445414912*x0**9*x1**43 + 12973869398532652032*x0**10*x1**42 + 9041246906589881344*x0**11*x1**41 + 13756551536919323648*x0**12*x1**40 + 16714084323800762368*x0**13*x1**39 + 878619958744909824*x0**14*x1**38 + 7445170964688148480*x0**15*x1**37 + 7021953997917218816*x0**16*x1**36 + 15517566368456784384*x0**17*x1**35 + 9747756523825580544*x0**18*x1**34 + 13048611595217106432*x0**19*x1**33 + 11213788980097373696*x0**20*x1**32 + 5038471379042734080*x0**21*x1**31 + 14780004738373193728*x0**22*x1**30 + 9058164226456662016*x0**23*x1**29 + 16215882475673022464*x0**24*x1**28 + 7795087274013214720*x0**25*x1**27 + 18325110153577170944*x0**26*x1**26 + 15677285570796271616*x0**27*x1**25 + 17586152276285798400*x0**28*x1**24 + 13369677521353854976*x0**29*x1**23 + 13458140851358846976*x0**30*x1**22 + 12037797419855867904*x0**31*x1**21 + 3541142233840861184*x0**32*x1**20 + 1732273089797179904*x0**33*x1**19 + 857814349612014080*x0**34*x1**18 + 15969846739556009472*x0**35*x1**17 + 6038834153750177280*x0**36*x1**16 + 1379438031917410304*x0**37*x1**15 + 11902654271792097280*x0**38*x1**14 + 1129634463681099776*x0**39*x1**13 + 3336189676906190848*x0**40*x1**12 + 18273465517453597696*x0**41*x1**11 + 2279474536374025216*x0**42*x1**10 + 13729645999818236928*x0**43*x1**9 + 4013036982641026048*x0**44*x1**8 + 6276835019811645440*x0**45*x1**7 + 1013062565126547456*x0**46*x1**6 + 13859598794612897792*x0**47*x1**5 + 204034444908111872*x0**48*x1**4 + 3784314577584276992*x0**49*x1**3 + 12233754794414945792*x0**50*x1**2 + 4084176828148402688*x0**51*x1 + 6383698479282776576*x0**52 + 12117844242406771712*x1**53 + 15317993628901147648*x0*x1**52 + 9651479782430657024*x0**2*x1**51 + 18256865819919818240*x0**3*x1**50 + 8256099953500156416*x0**4*x1**49 + 13999603752546887168*x0**5*x1**48 + 2952920618529880064*x0**6*x1**47 + 1628199537236961280*x0**7*x1**46 + 2835090223761108992*x0**8*x1**45 + 7203704519865864192*x0**9*x1**44 + 16050685484979184640*x0**10*x1**43 + 6198156401452493824*x0**11*x1**42 + 3284622485785379840*x0**12*x1**41 + 1787365197350251520*x0**13*x1**40 + 7888962150549436416*x0**14*x1**39 + 10658744266570672128*x0**15*x1**38 + 5590602106151439360*x0**16*x1**37 + 1122959419333149696*x0**17*x1**36 + 11066942701592184320*x0**18*x1**35 + 9749718091836772864*x0**19*x1**34 + 1738194865356910080*x0**20*x1**33 + 1831378866478381568*x0**21*x1**32 + 13460325608700866560*x0**22*x1**31 + 4008371201457213440*x0**23*x1**30 + 1622778028422529024*x0**24*x1**29 + 5685338022421987328*x0**25*x1**28 + 8706319563997067264*x0**26*x1**27 + 17051821716160292864*x0**27*x1**26 + 16397606005964683264*x0**28*x1**25 + 6980227749681391616*x0**29*x1**24 + 13888716557359362048*x0**30*x1**23 + 14086390016611848192*x0**31*x1**22 + 5854729236732453888*x0**32*x1**21 + 6550714423760292864*x0**33*x1**20 + 11498469744003020288*x0**34*x1**19 + 2874600111419564544*x0**35*x1**18 + 11220398849963191808*x0**36*x1**17 + 219838165357181440*x0**37*x1**16 + 15137619839678093312*x0**38*x1**15 + 10105359250667784192*x0**39*x1**14 + 8340138728613924864*x0**40*x1**13 + 13247048194778361856*x0**41*x1**12 + 307372297004393472*x0**42*x1**11 + 14270432856523697152*x0**43*x1**10 + 7130288510494016512*x0**44*x1**9 + 12754800036096453632*x0**45*x1**8 + 15573971592486262784*x0**46*x1**7 + 9851838504708065280*x0**47*x1**6 + 7060935696755672064*x0**48*x1**5 + 4286784505040497664*x0**49*x1**4 + 4636258740096694784*x0**50*x1**3 + 709991068069473792*x0**51*x1**2 + 4279702039930089984*x0**52*x1 + 16110830330998924800*x0**53 + 15516448274719717376*x1**54 + 11973796522583417344*x0*x1**53 + 6855131672815901184*x0**2*x1**52 + 17573440259103499264*x0**3*x1**51 + 12003616460160490496*x0**4*x1**50 + 13916266537514807808*x0**5*x1**49 + 9924698675374213632*x0**6*x1**48 + 11440579967450468352*x0**7*x1**47 + 11140545751224033280*x0**8*x1**46 + 3570552494841916416*x0**9*x1**45 + 10300929541122538496*x0**10*x1**44 + 1125576472193697792*x0**11*x1**43 + 6648951076807972864*x0**12*x1**42 + 13853591450935180288*x0**13*x1**41 + 16046869371295796224*x0**14*x1**40 + 8859837291037724672*x0**15*x1**39 + 3722334665359800320*x0**16*x1**38 + 16707075640655824384*x0**17*x1**37 + 3526033886728252928*x0**18*x1**36 + 6820596211984653312*x0**19*x1**35 + 14118501000066483200*x0**20*x1**34 + 12191654490483197440*x0**21*x1**33 + 12400173942886043136*x0**22*x1**32 + 12986715948469297152*x0**23*x1**31 + 1321414139680096256*x0**24*x1**30 + 7632156210086545408*x0**25*x1**29 + 18024065220233455616*x0**26*x1**28 + 6471252256398012416*x0**27*x1**27 + 6020031282725048320*x0**28*x1**26 + 9951244852547921920*x0**29*x1**25 + 9155709960959592448*x0**30*x1**24 + 8212340278139854848*x0**31*x1**23 + 13039687681638443008*x0**32*x1**22 + 17010314651055771136*x0**33*x1**21 + 9006877593310090752*x0**34*x1**20 + 14089613953141218304*x0**35*x1**19 + 1903933498968253440*x0**36*x1**18 + 17463094779105189376*x0**37*x1**17 + 9046924683292198400*x0**38*x1**16 + 5869302666226864128*x0**39*x1**15 + 10125940563875815424*x0**40*x1**14 + 1186821760451234816*x0**41*x1**13 + 17599051447186529280*x0**42*x1**12 + 15494979511353313280*x0**43*x1**11 + 17898582906669864960*x0**44*x1**10 + 9760510484438080512*x0**45*x1**9 + 10229857006998465536*x0**46*x1**8 + 3980076059083362304*x0**47*x1**7 + 9861868464379226112*x0**48*x1**6 + 14492120421150280192*x0**49*x1**5 + 12073423340360983040*x0**50*x1**4 + 13037449861686027264*x0**51*x1**3 + 7498367371023006720*x0**52*x1**2 + 17171101138021771776*x0**53*x1 + 17832092690276819456*x0**54 + 1943996458426704896*x1**55 + 2716488172733861888*x0*x1**54 + 11043113674191409664*x0**2*x1**53 + 16116635491787970048*x0**3*x1**52 + 6094683809246279680*x0**4*x1**51 + 8416954291674196992*x0**5*x1**50 + 2094299115768527360*x0**6*x1**49 + 15972737947588990464*x0**7*x1**48 + 8559639369668044800*x0**8*x1**47 + 3562857678931103744*x0**9*x1**46 + 11149174416981027840*x0**10*x1**45 + 3695230191039947776*x0**11*x1**44 + 18104776048581304320*x0**12*x1**43 + 9367493489784674304*x0**13*x1**42 + 11437679416169139200*x0**14*x1**41 + 5958646571260998656*x0**15*x1**40 + 17995688061560880128*x0**16*x1**39 + 6709680137814319104*x0**17*x1**38 + 1044510420349924864*x0**18*x1**37 + 1673779887977456128*x0**19*x1**36 + 13056421945594411008*x0**20*x1**35 + 1483256385035539456*x0**21*x1**34 + 4375045561039044096*x0**22*x1**33 + 375362161021833728*x0**23*x1**32 + 4975843814020067328*x0**24*x1**31 + 14781257334635339776*x0**25*x1**30 + 13865256630036891648*x0**26*x1**29 + 1905173425297512448*x0**27*x1**28 + 12968118012915810304*x0**28*x1**27 + 6136618654258745344*x0**29*x1**26 + 6733038099008456704*x0**30*x1**25 + 470940436380153856*x0**31*x1**24 + 6087622274853854208*x0**32*x1**23 + 3039599144048119808*x0**33*x1**22 + 7093236953567190528*x0**34*x1**21 + 5477137165209926144*x0**35*x1**20 + 9216997940494002176*x0**36*x1**19 + 14800305224209830912*x0**37*x1**18 + 9681606594076579328*x0**38*x1**17 + 5725137323764767232*x0**39*x1**16 + 6428466128745113600*x0**40*x1**15 + 13139763461059530752*x0**41*x1**14 + 2386859990435679232*x0**42*x1**13 + 9124815926833529856*x0**43*x1**12 + 16390491244007497728*x0**44*x1**11 + 11498563566002657280*x0**45*x1**10 + 13746848797856574464*x0**46*x1**9 + 7855661852769571840*x0**47*x1**8 + 154344803511895040*x0**48*x1**7 + 1032387611754463232*x0**49*x1**6 + 4575356899928119808*x0**50*x1**5 + 17888248964091301376*x0**51*x1**4 + 6538496483793577984*x0**52*x1**3 + 16874053734501428224*x0**53*x1**2 + 9760216984182469120*x0**54*x1 + 13256299643833155072*x0**55 + 17258954060319342592*x1**56 + 244828446531794432*x0*x1**55 + 784240221957043712*x0**2*x1**54 + 2089888615284811264*x0**3*x1**53 + 4591295013270451712*x0**4*x1**52 + 18063870275627137536*x0**5*x1**51 + 10091204516137015808*x0**6*x1**50 + 12603658556987480576*x0**7*x1**49 + 12290317317571045888*x0**8*x1**48 + 13004663069918868480*x0**9*x1**47 + 14620358278513964032*x0**10*x1**46 + 15509127179590669312*x0**11*x1**45 + 4646462837701112832*x0**12*x1**44 + 14502884516134714368*x0**13*x1**43 + 2177855863264295936*x0**14*x1**42 + 896867343200975872*x0**15*x1**41 + 8817483626029587456*x0**16*x1**40 + 7135285429146341888*x0**17*x1**39 + 5287628037452517888*x0**18*x1**38 + 8679173764780450304*x0**19*x1**37 + 3825563676264321536*x0**20*x1**36 + 4166743078889762304*x0**21*x1**35 + 6902131450398821888*x0**22*x1**34 + 7922419606484585984*x0**23*x1**33 + 14142189898106371584*x0**24*x1**32 + 14617991886799337472*x0**25*x1**31 + 14191786985819764736*x0**26*x1**30 + 4278433899275728896*x0**27*x1**29 + 15816613236535130112*x0**28*x1**28 + 9538738386587543552*x0**29*x1**27 + 4100273013887821824*x0**30*x1**26 + 2563121146904635392*x0**31*x1**25 + 2594067892332292096*x0**32*x1**24 + 13217172845933394432*x0**33*x1**23 + 3720548893094640128*x0**34*x1**22 + 646497006961592832*x0**35*x1**21 + 9079472126507563520*x0**36*x1**20 + 659963883754485248*x0**37*x1**19 + 9236873388377511424*x0**38*x1**18 + 13258791424666670592*x0**39*x1**17 + 11327533184614254080*x0**40*x1**16 + 1621880736279270400*x0**41*x1**15 + 2946966208980194304*x0**42*x1**14 + 15336195653501023232*x0**43*x1**13 + 3698137519519650816*x0**44*x1**12 + 5505045912865627136*x0**45*x1**11 + 5061814874197203968*x0**46*x1**10 + 13699277634491833344*x0**47*x1**9 + 10517755502877342720*x0**48*x1**8 + 2230816681775644160*x0**49*x1**7 + 17958988432453844480*x0**50*x1**6 + 4700412625681690112*x0**51*x1**5 + 433388630330490368*x0**52*x1**4 + 12951770286559519232*x0**53*x1**3 + 2840748815803007488*x0**54*x1**2 + 1715408989899031040*x0**55*x1 + 2400275382198014464*x0**56 + 1399786055761402880*x1**57 + 13775749287407803392*x0*x1**56 + 16476517695262007808*x0**2*x1**55 + 35927288533659136*x0**3*x1**54 + 4078838070541525504*x0**4*x1**53 + 10261344022328155648*x0**5*x1**52 + 15737810590567169536*x0**6*x1**51 + 9193205179895099904*x0**7*x1**50 + 150296590995069440*x0**8*x1**49 + 12006766918235471360*x0**9*x1**48 + 10535426834079898624*x0**10*x1**47 + 17183404858941834240*x0**11*x1**46 + 12346631499397282816*x0**12*x1**45 + 7433193727466898432*x0**13*x1**44 + 17399302106145526784*x0**14*x1**43 + 5667772882170186752*x0**15*x1**42 + 2845685281576075264*x0**16*x1**41 + 6611754393067657216*x0**17*x1**40 + 16852088281402029568*x0**18*x1**39 + 12517623678021122560*x0**19*x1**38 + 3338714965205374464*x0**20*x1**37 + 7902423703176849920*x0**21*x1**36 + 10343632997603719680*x0**22*x1**35 + 7924116540874315264*x0**23*x1**34 + 1746172042984285696*x0**24*x1**33 + 16569737840532673024*x0**25*x1**32 + 5205514113929635840*x0**26*x1**31 + 9616492682223114240*x0**27*x1**30 + 10543727570908588032*x0**28*x1**29 + 274494694422075392*x0**29*x1**28 + 5077106041734526976*x0**30*x1**27 + 15361079332470999040*x0**31*x1**26 + 6819692327928466432*x0**32*x1**25 + 11120542536533171200*x0**33*x1**24 + 16211163124933783040*x0**34*x1**23 + 6176314973385474560*x0**35*x1**22 + 14687929922292755968*x0**36*x1**21 + 3295519614757245440*x0**37*x1**20 + 12068007798541292032*x0**38*x1**19 + 3075274670671871488*x0**39*x1**18 + 4602097433046984192*x0**40*x1**17 + 18231958425705964032*x0**41*x1**16 + 4406735740329262080*x0**42*x1**15 + 8332984375859700736*x0**43*x1**14 + 17669014319896681472*x0**44*x1**13 + 2932175874417648640*x0**45*x1**12 + 12814204970761106432*x0**46*x1**11 + 12421156414261914624*x0**47*x1**10 + 14452638323816880128*x0**48*x1**9 + 13210118236265279488*x0**49*x1**8 + 17878971684189993472*x0**50*x1**7 + 5809524211480040960*x0**51*x1**6 + 14101239047808883200*x0**52*x1**5 + 10121998215961765376*x0**53*x1**4 + 8662314447194551808*x0**54*x1**3 + 15238867771772729856*x0**55*x1**2 + 7473273263663290880*x0**56*x1 + 5176266717947776512*x0**57 + 18286351715494348800*x1**58 + 8777953281465940480*x0*x1**57 + 11849823564808102400*x0**2*x1**56 + 14797041520099268608*x0**3*x1**55 + 5804352941402482688*x0**4*x1**54 + 10847560183775327232*x0**5*x1**53 + 7826013979918943232*x0**6*x1**52 + 14067897207903197184*x0**7*x1**51 + 12946334925317173248*x0**8*x1**50 + 4001730989153127936*x0**9*x1**49 + 8856784690534499840*x0**10*x1**48 + 8040068300675125248*x0**11*x1**47 + 6122657807411228672*x0**12*x1**46 + 15503911509187725312*x0**13*x1**45 + 2945175842465425408*x0**14*x1**44 + 1660771390239133696*x0**15*x1**43 + 8702867283428890624*x0**16*x1**42 + 8484415791654824448*x0**17*x1**41 + 4362949530483905024*x0**18*x1**40 + 15304443464581138432*x0**19*x1**39 + 5138092627675543552*x0**20*x1**38 + 11266798686800808960*x0**21*x1**37 + 2413592654156839936*x0**22*x1**36 + 3338573375558027264*x0**23*x1**35 + 11950439958550536192*x0**24*x1**34 + 4511545481719770624*x0**25*x1**33 + 12387248664228275712*x0**26*x1**32 + 6139395944322392064*x0**27*x1**31 + 5502664813429727232*x0**28*x1**30 + 6140952140902133760*x0**29*x1**29 + 17767912354355949568*x0**30*x1**28 + 3515287471036030976*x0**31*x1**27 + 14217283771437185024*x0**32*x1**26 + 6225905259446754816*x0**33*x1**25 + 16126852977973040640*x0**34*x1**24 + 7871596951642187776*x0**35*x1**23 + 8114700677716400128*x0**36*x1**22 + 8098501878407318528*x0**37*x1**21 + 14921112642009463808*x0**38*x1**20 + 8775915705574959104*x0**39*x1**19 + 15735834688119271424*x0**40*x1**18 + 18165294811043682816*x0**41*x1**17 + 10493849197239321088*x0**42*x1**16 + 16039494078668541952*x0**43*x1**15 + 1052765917983748096*x0**44*x1**14 + 830142799437793280*x0**45*x1**13 + 1850048386983241728*x0**46*x1**12 + 10904762151559581696*x0**47*x1**11 + 16884138875521558528*x0**48*x1**10 + 10166341536918879744*x0**49*x1**9 + 8168055003356427776*x0**50*x1**8 + 8156612657307023360*x0**51*x1**7 + 3354579367011108864*x0**52*x1**6 + 7229009512526132224*x0**53*x1**5 + 16932907077860459520*x0**54*x1**4 + 10221270923970459648*x0**55*x1**3 + 11547671123936493568*x0**56*x1**2 + 9171222026356435456*x0**57*x1 + 12884842118543233536*x0**58 + 10345373075465307136*x1**59 + 10088630513307277312*x0*x1**58 + 5472363930022068736*x0**2*x1**57 + 563235863832555008*x0**3*x1**56 + 1413203382101606400*x0**4*x1**55 + 1770277996110731264*x0**5*x1**54 + 863301236063163392*x0**6*x1**53 + 8575233410413173760*x0**7*x1**52 + 11156186759947961344*x0**8*x1**51 + 5703597158916157440*x0**9*x1**50 + 9085893955719558656*x0**10*x1**49 + 17407986269345753600*x0**11*x1**48 + 2830166816397729792*x0**12*x1**47 + 7089722088397131776*x0**13*x1**46 + 4692813092922066944*x0**14*x1**45 + 17229220395071444992*x0**15*x1**44 + 5362854473720624128*x0**16*x1**43 + 7305789537421838336*x0**17*x1**42 + 1974946062432686592*x0**18*x1**41 + 1659603398580583936*x0**19*x1**40 + 4054542145804623872*x0**20*x1**39 + 1605183631016839168*x0**21*x1**38 + 1964684638590069760*x0**22*x1**37 + 14915238349616964608*x0**23*x1**36 + 12129319929493238784*x0**24*x1**35 + 12455739004271951872*x0**25*x1**34 + 11092589703817360896*x0**26*x1**33 + 7521135623223167488*x0**27*x1**32 + 16307028652923240448*x0**28*x1**31 + 17244773088212279296*x0**29*x1**30 + 17777804389808697344*x0**30*x1**29 + 4855652205583486976*x0**31*x1**28 + 2819824323655945216*x0**32*x1**27 + 12383380365701343232*x0**33*x1**26 + 4562416453903234560*x0**34*x1**25 + 7485810080347691520*x0**35*x1**24 + 8345357381754667008*x0**36*x1**23 + 17885941389284534272*x0**37*x1**22 + 8896215545217149952*x0**38*x1**21 + 7795867462398059520*x0**39*x1**20 + 3371944921229921280*x0**40*x1**19 + 18010503852726595584*x0**41*x1**18 + 12358709392933491200*x0**42*x1**17 + 2911063912396885504*x0**43*x1**16 + 10598775923100524544*x0**44*x1**15 + 13047940763339214848*x0**45*x1**14 + 16207116206717798400*x0**46*x1**13 + 6209205902504695808*x0**47*x1**12 + 1451832683330976768*x0**48*x1**11 + 15379258435000334336*x0**49*x1**10 + 15289446308607145472*x0**50*x1**9 + 1781896603798026752*x0**51*x1**8 + 11728464888250064896*x0**52*x1**7 + 5075950735836280832*x0**53*x1**6 + 5056428338189521920*x0**54*x1**5 + 2640690249708297216*x0**55*x1**4 + 8668111400684059648*x0**56*x1**3 + 2429081833661587456*x0**57*x1**2 + 13929956945419665920*x0**58*x1 + 9226806911908838912*x0**59 + 17222747938460009984*x0*x1**59 + 1926208032420922880*x0**2*x1**58 + 17282702108451226112*x0**3*x1**57 + 13438037599234482688*x0**4*x1**56 + 8256580232328389632*x0**5*x1**55 + 10606231845388100608*x0**6*x1**54 + 4788329614019959808*x0**7*x1**53 + 5278377364344822784*x0**8*x1**52 + 7815000640369428992*x0**9*x1**51 + 1281352428848122368*x0**10*x1**50 + 10377290002811264512*x0**11*x1**49 + 5272845951686639104*x0**12*x1**48 + 15130239771709802496*x0**13*x1**47 + 13420374001305913344*x0**14*x1**46 + 14922994910924322816*x0**15*x1**45 + 1366983759232153600*x0**16*x1**44 + 450988911199413760*x0**17*x1**43 + 651008717851878912*x0**18*x1**42 + 2029921936380363264*x0**19*x1**41 + 1759996776674173440*x0**20*x1**40 + 8309468291248307200*x0**21*x1**39 + 2177103814246087680*x0**22*x1**38 + 18311142247252640768*x0**23*x1**37 + 17817145739681891328*x0**24*x1**36 + 5472692317070400000*x0**25*x1**35 + 9971812460673473024*x0**26*x1**34 + 10313362006578703872*x0**27*x1**33 + 17856615575134375424*x0**28*x1**32 + 18425911784293265408*x0**29*x1**31 + 5954935479182290944*x0**30*x1**30 + 2853103920883077120*x0**31*x1**29 + 6745490302339092480*x0**32*x1**28 + 17732182981518295552*x0**33*x1**27 + 10506225473413186048*x0**34*x1**26 + 16362020767967729152*x0**35*x1**25 + 2560087218080036352*x0**36*x1**24 + 2106552092931021824*x0**37*x1**23 + 11951534000487095296*x0**38*x1**22 + 16709953972691362816*x0**39*x1**21 + 12798541247420408832*x0**40*x1**20 + 12978731655789596160*x0**41*x1**19 + 17686510945031212544*x0**42*x1**18 + 2330224345411168768*x0**43*x1**17 + 16714719240344128000*x0**44*x1**16 + 15917700715159586816*x0**45*x1**15 + 10354282247052036096*x0**46*x1**14 + 17450926402912696320*x0**47*x1**13 + 8034796488298608640*x0**48*x1**12 + 367085128961397248*x0**49*x1**11 + 12055210688754913792*x0**50*x1**10 + 215165695607334400*x0**51*x1**9 + 7379478710309963264*x0**52*x1**8 + 1083314783094293504*x0**53*x1**7 + 4519018995033359360*x0**54*x1**6 + 10182551190556589056*x0**55*x1**5 + 6505290961304484864*x0**56*x1**4 + 16304989953836781056*x0**57*x1**3 + 16065399798595128832*x0**58*x1**2 + 3444028858934397440*x0**59*x1 + 5598801169564740096*x0**60 + 14608307199701680128*x0*x1**60 + 7709652388663001600*x0**2*x1**59 + 3279412177097750016*x0**3*x1**58 + 11899961570862390784*x0**4*x1**57 + 15804611035110701568*x0**5*x1**56 + 4870783734722571264*x0**6*x1**55 + 12080325460845159424*x0**7*x1**54 + 12947233225248048128*x0**8*x1**53 + 875706210610711552*x0**9*x1**52 + 6678329209340513792*x0**10*x1**51 + 15239519069575756288*x0**11*x1**50 + 14378609607335846400*x0**12*x1**49 + 14659397464832034304*x0**13*x1**48 + 17844538588466583552*x0**14*x1**47 + 10151391413808596992*x0**15*x1**46 + 12538672940900583424*x0**16*x1**45 + 2581824629061163008*x0**17*x1**44 + 18413812609582096896*x0**18*x1**43 + 3173411710797190656*x0**19*x1**42 + 11222615481947938304*x0**20*x1**41 + 7057965143072471552*x0**21*x1**40 + 17950582641130009600*x0**22*x1**39 + 1632327026632784896*x0**23*x1**38 + 2630185231834356736*x0**24*x1**37 + 10219892923034293248*x0**25*x1**36 + 13804316479035489792*x0**26*x1**35 + 9780772731991394816*x0**27*x1**34 + 9069793660303135232*x0**28*x1**33 + 17400341310982575616*x0**29*x1**32 + 17645832449921372160*x0**30*x1**31 + 13575752143617126400*x0**31*x1**30 + 17372286629490454528*x0**32*x1**29 + 12292853493024149504*x0**33*x1**28 + 9277261555159520768*x0**34*x1**27 + 7830239435520561664*x0**35*x1**26 + 6722608682681272832*x0**36*x1**25 + 1244012413787520512*x0**37*x1**24 + 11791567829126140928*x0**38*x1**23 + 6721248096463297536*x0**39*x1**22 + 5926051874493398016*x0**40*x1**21 + 6192570220437632000*x0**41*x1**20 + 13007860166596262400*x0**42*x1**19 + 9564507966796619264*x0**43*x1**18 + 15091686747330942464*x0**44*x1**17 + 10680909948038144512*x0**45*x1**16 + 3326209394228303872*x0**46*x1**15 + 14967379777448478720*x0**47*x1**14 + 17700691026676746240*x0**48*x1**13 + 16923990070297171968*x0**49*x1**12 + 17473775597008406016*x0**50*x1**11 + 14597060577542382080*x0**51*x1**10 + 6003507047048067584*x0**52*x1**9 + 13726882639617150464*x0**53*x1**8 + 6242270733180734464*x0**54*x1**7 + 2974821090994127872*x0**55*x1**6 + 482113861110490112*x0**56*x1**5 + 4987857533824904192*x0**57*x1**4 + 18430647223495555584*x0**58*x1**3 + 3559039974274588160*x0**59*x1**2 + 3587935139851695616*x0**60*x1 + 16763574290514706944*x0**61 + 17600623696647553024*x0**2*x1**60 + 4938126622667571200*x0**3*x1**59 + 18095107061007253504*x0**4*x1**58 + 4674824374140796928*x0**5*x1**57 + 8747562777981222912*x0**6*x1**56 + 3245758325194752000*x0**7*x1**55 + 9972684813137608704*x0**8*x1**54 + 16576518775327686656*x0**9*x1**53 + 7087788403202195456*x0**10*x1**52 + 5917448435388121088*x0**11*x1**51 + 8916691855588982784*x0**12*x1**50 + 14316644148248051712*x0**13*x1**49 + 6760287869752836096*x0**14*x1**48 + 12862702748235202560*x0**15*x1**47 + 2160584329044951040*x0**16*x1**46 + 14346286981732892672*x0**17*x1**45 + 2110906194678775808*x0**18*x1**44 + 16632426742576840704*x0**19*x1**43 + 15293903277154893824*x0**20*x1**42 + 172772859142209536*x0**21*x1**41 + 820849201809195008*x0**22*x1**40 + 5966073237614886912*x0**23*x1**39 + 6881473842343051264*x0**24*x1**38 + 9858485137430282240*x0**25*x1**37 + 5851400771907616768*x0**26*x1**36 + 14519745936130834432*x0**27*x1**35 + 7315591819314593792*x0**28*x1**34 + 6424367276257968128*x0**29*x1**33 + 4578689674477830144*x0**30*x1**32 + 477663035477983232*x0**31*x1**31 + 2593827094760783872*x0**32*x1**30 + 11683322595817553920*x0**33*x1**29 + 11053889572311138304*x0**34*x1**28 + 16342507516564865024*x0**35*x1**27 + 10221433925759139840*x0**36*x1**26 + 5359406701873201152*x0**37*x1**25 + 13590071471450882048*x0**38*x1**24 + 4746442163527614464*x0**39*x1**23 + 16521569582217953280*x0**40*x1**22 + 16695512321732116480*x0**41*x1**21 + 10154146013162504192*x0**42*x1**20 + 2931280407464771584*x0**43*x1**19 + 604181639462912000*x0**44*x1**18 + 14046252248745377792*x0**45*x1**17 + 10580708146227970048*x0**46*x1**16 + 6153465203360006144*x0**47*x1**15 + 9737327752142389248*x0**48*x1**14 + 1670905830498631680*x0**49*x1**13 + 8705112833056047104*x0**50*x1**12 + 14317435796620050432*x0**51*x1**11 + 6975112650610966528*x0**52*x1**10 + 16368737465957089280*x0**53*x1**9 + 15570925231773057024*x0**54*x1**8 + 4594304938615504896*x0**55*x1**7 + 1311840517239078912*x0**56*x1**6 + 7711745858802286592*x0**57*x1**5 + 6395250009331204096*x0**58*x1**4 + 15481827406527856640*x0**59*x1**3 + 12525878356182302720*x0**60*x1**2 + 7668170013970268160*x0**61*x1 + 17965740721906384896*x0**62 + 7882224037177327616*x0**2*x1**61 + 12590403196058337280*x0**3*x1**60 + 18438306421477998592*x0**4*x1**59 + 5368095042755887104*x0**5*x1**58 + 1283568774754074624*x0**6*x1**57 + 18139600998048464896*x0**7*x1**56 + 16336943987728318464*x0**8*x1**55 + 18357564884603895808*x0**9*x1**54 + 1327995641585991680*x0**10*x1**53 + 2788604480109674496*x0**11*x1**52 + 1452993620989706240*x0**12*x1**51 + 1918122223911043072*x0**13*x1**50 + 15305776903223246848*x0**14*x1**49 + 1064823135431294976*x0**15*x1**48 + 6539464153453559808*x0**16*x1**47 + 10040555466896965632*x0**17*x1**46 + 1590532630019833856*x0**18*x1**45 + 777263461372526592*x0**19*x1**44 + 18103931741331783680*x0**20*x1**43 + 13889626817368686592*x0**21*x1**42 + 8606074323184123904*x0**22*x1**41 + 989335065114705920*x0**23*x1**40 + 9621945001923575808*x0**24*x1**39 + 10404955610574487552*x0**25*x1**38 + 2036657273966690304*x0**26*x1**37 + 1261264081873010688*x0**27*x1**36 + 2384148028320645120*x0**28*x1**35 + 18107120325052334080*x0**29*x1**34 + 8598619634347802624*x0**30*x1**33 + 6816874235676327936*x0**31*x1**32 + 7031755091627474944*x0**32*x1**31 + 14170875294684020736*x0**33*x1**30 + 11907670246883852288*x0**34*x1**29 + 2266360646171885568*x0**35*x1**28 + 12542562345622175744*x0**36*x1**27 + 3867248476702113792*x0**37*x1**26 + 6328090739595018240*x0**38*x1**25 + 9572573631301550080*x0**39*x1**24 + 2752477826555838464*x0**40*x1**23 + 15165330785449279488*x0**41*x1**22 + 7109643395827499008*x0**42*x1**21 + 13988927011008020480*x0**43*x1**20 + 6532477856570671104*x0**44*x1**19 + 5076592519999913984*x0**45*x1**18 + 320794612031553536*x0**46*x1**17 + 10712191045212307456*x0**47*x1**16 + 7071170384459988992*x0**48*x1**15 + 12266178107748122624*x0**49*x1**14 + 2882927184610066432*x0**50*x1**13 + 3058937005984448512*x0**51*x1**12 + 13557455558524534784*x0**52*x1**11 + 5140183874503966720*x0**53*x1**10 + 9831156775921909760*x0**54*x1**9 + 14389001908960362496*x0**55*x1**8 + 10616176192316768256*x0**56*x1**7 + 16530018229565784064*x0**57*x1**6 + 5146227889921851392*x0**58*x1**5 + 11438741731776921600*x0**59*x1**4 + 14592144378773372928*x0**60*x1**3 + 15718229003569463296*x0**61*x1**2 + 2075760305496915968*x0**62*x1 + 825562808157470720*x0**63 + 17552645407257919488*x0**3*x1**61 + 800213567579095040*x0**4*x1**60 + 5559662953663299584*x0**5*x1**59 + 10957403028927283200*x0**6*x1**58 + 2085019292914417664*x0**7*x1**57 + 12701173494998630400*x0**8*x1**56 + 7863821511063240704*x0**9*x1**55 + 9082185947755315200*x0**10*x1**54 + 15187894963074498560*x0**11*x1**53 + 1305381985937522688*x0**12*x1**52 + 14166683956358938624*x0**13*x1**51 + 12018642855963656192*x0**14*x1**50 + 7035275727859613696*x0**15*x1**49 + 18003695863297212416*x0**16*x1**48 + 14292472484623024128*x0**17*x1**47 + 510894674915885056*x0**18*x1**46 + 15303880245665333248*x0**19*x1**45 + 2531680498535628800*x0**20*x1**44 + 14387105251402448896*x0**21*x1**43 + 14113546758411780096*x0**22*x1**42 + 9347029711584231424*x0**23*x1**41 + 680635080988688384*x0**24*x1**40 + 14653332200859107328*x0**25*x1**39 + 17628663442179096576*x0**26*x1**38 + 4853620357979963392*x0**27*x1**37 + 11610576707500638208*x0**28*x1**36 + 8981281566651056128*x0**29*x1**35 + 6176700083078168576*x0**30*x1**34 + 3547404942228586496*x0**31*x1**33 + 6581080668564881408*x0**32*x1**32 + 15628299947533664256*x0**33*x1**31 + 15628299947533664256*x0**34*x1**30 + 6581080668564881408*x0**35*x1**29 + 3547404942228586496*x0**36*x1**28 + 6176700083078168576*x0**37*x1**27 + 8981281566651056128*x0**38*x1**26 + 11610576707500638208*x0**39*x1**25 + 4853620357979963392*x0**40*x1**24 + 17628663442179096576*x0**41*x1**23 + 14653332200859107328*x0**42*x1**22 + 680635080988688384*x0**43*x1**21 + 9347029711584231424*x0**44*x1**20 + 14113546758411780096*x0**45*x1**19 + 14387105251402448896*x0**46*x1**18 + 2531680498535628800*x0**47*x1**17 + 15303880245665333248*x0**48*x1**16 + 510894674915885056*x0**49*x1**15 + 14292472484623024128*x0**50*x1**14 + 18003695863297212416*x0**51*x1**13 + 7035275727859613696*x0**52*x1**12 + 12018642855963656192*x0**53*x1**11 + 14166683956358938624*x0**54*x1**10 + 1305381985937522688*x0**55*x1**9 + 15187894963074498560*x0**56*x1**8 + 9082185947755315200*x0**57*x1**7 + 7863821511063240704*x0**58*x1**6 + 12701173494998630400*x0**59*x1**5 + 2085019292914417664*x0**60*x1**4 + 10957403028927283200*x0**61*x1**3 + 5559662953663299584*x0**62*x1**2 + 800213567579095040*x0**63*x1 + 17552645407257919488*x0**64)
]

if __name__ == '__main__':
  # Parse the arguments
  parser = argparse.ArgumentParser(description="Efficient Normalized Reduction and Generation of Equivalent Multivariate Binary Polynomials")
  parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
  arguments = parser.parse_args()
  # Run the polynomial tests
  run_polynomial_tests(arguments.debug)
  # Univariate examples
  for w, p in univariate_polynomials:
    print("="*200)
    p = p.as_polynomial()
    print(f"p: {p}")
    q = normalize_univariate_polynomial(p, w, arguments.debug)
    print(f"q = normalize_univariate_polynomial(p): {q}")
    e = equivalent_univariate_polynomial(q, w, q.degree() + randrange(5) + 1, arguments.debug)
    print(f"e = equivalent_univariate_polynomial(q): {e}")
    n = normalize_univariate_polynomial(e, w, arguments.debug)
    print(f"n = normalize_univariate_polynomial(e): {n}")
    assert n == q
  # Multivariate examples
  for w, p in multivariate_polynomials:
    print("="*200)
    p = p.as_polynomial()
    print(f"p: {p}")
    q = normalize_multivariate_polynomial(p, w, arguments.debug)
    print(f"q = normalize_multivariate_polynomial(p): {q}")
    e = equivalent_multivariate_polynomial(q, w, q.degree() + randrange(5) + 1, arguments.debug)
    print(f"e = equivalent_multivariate_polynomial(q): {e}")
    n = normalize_multivariate_polynomial(e, w, arguments.debug)
    print(f"n = normalize_multivariate_polynomial(e): {n}")
    assert n == q
  # Notify the end
  print("="*200)
