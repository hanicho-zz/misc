#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:12:05 2018

TODOs
* Make functions bound to real numbers
* Check for any weird pointer re-writes

WARNING
This code calls sympify, which uses eval()
"""

import sys

import copy
from functools import reduce
from collections import OrderedDict
import inspect

import numpy as np
from sympy import Symbol, solveset, simplify, Function, sympify, Id, sin, S, EmptySet

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, Iterator, Generic, TypeVar, Dict, Text, Mapping, Optional, List, Sequence, Set, Tuple, Union

Real     = float
Variable = Text
T        = TypeVar('T')

# Need to create Sympy fuzzy sets
# Need to make sure constraints/invariants are within the proper scope
# are constraints part of scope?
# ‹ ›  « »

class Indented(ABC):
    def __indent__(self, depth: int=0) -> Text:
        pass

def indent(p: Indented, depth: int=0) -> Text:
    return p.__indent__(depth)

class Scoped(ABC):
    @abstractproperty
    def scope(self) -> Set[Variable]:
        pass

class Expression(Indented, Scoped):
    def __init__(self,
                 expr: Any,
                 args: Tuple[Union['Expression', Real, Variable], ...]=()) -> None:
        expr = copy.deepcopy(expr)
        args = copy.deepcopy(args)

        try:
            expr = sympify(expr)

            if expr is None:
                raise Exception
        except Exception:
            raise TypeError('Unable to sympify {0}'.format(expr))

        assert isinstance(args, tuple)

        self._expr = expr
        self._args = []

        for arg in args:
            if isinstance(arg, (Expression, Real, Text)):
                self._args.append(arg)
            else:
                self._args.append(sympify(arg))

        self._args = tuple(args)

    def __indent__(self, depth: int=0) -> Text:
        return '{0}{1}({2})'.format(
            '\t'*depth,
            self.expr,
            ', '.join(map(lambda arg: str(arg), self.args))
        )

    def __repr__(self) -> Text:
        return indent(self)

    @property
    def expr(self) -> Any:
        return self._expr

    @property
    def args(self) -> Tuple[Union['Expression', Real, Variable], ...]:
        return self._args

    @property
    def scope(self) -> Set[Variable]:
        scope = set()

        for arg in self._args:
            if isinstance(arg, Expression):
                scope |= arg.scope
            elif isinstance(arg, Real):
                pass
            elif isinstance(arg, Variable):
                scope.add(arg)
            else:
                raise TypeError('{0}'.format(arg))

        return scope


# Check constraints before single eval
class Constrained(Scoped):
    @abstractproperty
    def constraints(self) -> Set[Expression]:
        pass

# Check constraints errwhere
class Invariant(Constrained):
    @abstractmethod
    def propagate(self, additions: Set[Expression]=set()) -> None:
        pass

    @abstractproperty
    def invariants(self) -> Set[Expression]:
        pass


class Assignment(Indented, Constrained):
    def __init__(self,
                 var:         Variable,
                 expr:        Expression,
                 constraints: Set[Expression]=set()) -> None:
        var         = copy.deepcopy(var)
        expr        = copy.deepcopy(expr)
        constraints = copy.deepcopy(constraints)

        self._var         = var
        self._expr        = expr
        self._constraints = constraints

    def __indent__(self, depth: int=0) -> Text:
        return '{0}{1} := {2}\t{3}'.format(
            '\t'*depth,
            self.var,
            self.expr,
            '‹{0}› '.format(' & '.join(str(constraint) for constraint in self._constraints))
        )

    def __repr__(self) -> Text:
        return indent(self)

    @property
    def var(self) -> Variable:
        return self._var

    @property
    def expr(self) -> Expression:
        return self._expr

    @property
    def scope(self) -> Set[Variable]:
        return (self._expr.scope | {self._var})

    @property
    def constraints(self) -> Set[Expression]:
        return self._constraints


class Block(Indented, Invariant, Sequence['Statement']):
    def __init__(self,
                 statements:  Tuple['Statement'],
                 constraints: Set[Expression]=set(),
                 invariants:  Set[Expression]=set()) -> None:
        statements  = copy.deepcopy(statements)
        constraints = copy.deepcopy(constraints)
        invariants  = copy.deepcopy(invariants)

        assert len(statements) > 0

        for statement in statements:
            assert isinstance(statement,
                              (Definition,
                               Loop,
                               Conditional,
                               Block,
                               Assignment))

        self._statements  = statements
        self._constraints = constraints
        self._invariants  = invariants

        self.propagate(self._invariants)

    def __getitem__(self, i: int) -> 'Statement':
        return self.statements[i]

    def __iter__(self) -> Iterator['Statement']:
        return iter(self.statements)

    def __len__(self) -> int:
        return len(self.statements)

    def __indent__(self, depth: int=0) -> Text:
        ret = ''

        for i in range(len(self.statements)):
            ret += indent(self.statements[i], depth)

            if i != len(self.statements)-1:
                ret += '\n'

            if (i > 0 and not
                  (isinstance(self.statements[i-1], Assignment) and
                   isinstance(self.statements[i],   Assignment))):
                ret += '\n'

        if depth == 0:
            ret += '\n'

        return ret

    def __repr__(self) -> Text:
        return indent(self)

    def propagate(self, additions: Set[Expression]=set()) -> None:
        # Down-propagate invariants
        self._constraints |= additions
        self._invariants  |= additions

        for statement in self._statements:
            if isinstance(statement, Assignment):
                statement._constraints |= self._invariants
            else:
                statement.propagate(self._invariants)

    @property
    def statements(self) -> Tuple['Statement']:
        return self._statements

    @property
    def scope(self) -> Set[Variable]:
        return reduce(lambda s1, s2: s1 | s2,
                      map(lambda statement: statement.scope,
                          self._statements))

    @property
    def constraints(self) -> Set[Expression]:
        return self._constraints

    @property
    def invariants(self) -> Set[Expression]:
        return self._invariants


class Branch(Indented, Invariant, Sequence['Statement']):
    def __init__(self,
                 condition:   Optional[Expression],
                 block:       Block,
                 constraints: Set[Expression]=set(),
                 invariants:  Set[Expression]=set()) -> None:
        condition   = copy.deepcopy(condition)
        block       = copy.deepcopy(block)
        constraints = copy.deepcopy(constraints)
        invariants  = copy.deepcopy(invariants)

        assert isinstance(condition, Expression) or condition is None
        assert isinstance(block, Block)
        assert isinstance(invariants, set)
        assert isinstance(constraints, set)

        self._condition   = condition
        self._block       = block
        self._constraints = constraints
        self._invariants  = invariants

        self.propagate(self._invariants)

    def __getitem__(self, i: int) -> 'Statement':
        return self.block[i]

    def __iter__(self) -> Iterator['Statement']:
        return iter(self.block)

    def __len__(self) -> int:
        return len(self.block)

    def __indent__(self, depth: int=0) -> Text:
        return indent(self.block, depth)

    def __repr__(self) -> Text:
        return indent(self)

    def propagate(self, additions: Set[Expression]=set()) -> None:
        # Down-propagate invariants
        self._constraints |= additions
        self._invariants  |= additions
        self._block.propagate(self._invariants)

    @property
    def condition(self) -> Optional[Expression]:
        return self._condition

    @property
    def block(self) -> Block:
        return self._block

    @property
    def scope(self) -> Set[Variable]:
        scope = set()

        if self._condition is not None:
            scope |= self._condition.scope

        scope |= self._block.scope

        return scope

    @property
    def constraints(self) -> Set[Expression]:
        return self._constraints

    @property
    def invariants(self) -> Set[Expression]:
        return self._invariants


class Conditional(Indented, Invariant, Mapping):
    def __init__(self,
                 branches:    Tuple[Branch]) -> None:
        branches = copy.deepcopy(branches)

        conditions = []

        for i in range(len(branches)):
            assert isinstance(branches[i], Branch)

            if i != len(branches)-1:
                assert branches[i].condition is not None

            assert (isinstance(branches[i].condition, Expression) or
                    branches[i].condition is None)

            conditions.append(branches[i].condition)

        self._branches = branches
        self._mapping  = dict(zip(conditions, branches))

    def __getitem__(self, condition: Optional[Expression]) -> Tuple[Block]:
        return self._mapping[condition]

    def __iter__(self) -> Iterator[Optional[Expression]]:
        return iter(self.branches)

    def __len__(self) -> int:
        return len(self.branches)

    def __indent__(self, depth: int=0) -> Text:
        ret = ''

        for i in range(len(self.branches)):
            if i == 0:
                ret += '\t'*depth
                ret += 'if '
                ret += indent(self.branches[i].condition, 0)
                ret += ' {\t'
                ret += '‹{0}› '.format(' & '.join(str(constraint) for constraint in self.branches[i].constraints))
                ret += '«{0}»'.format(' & '.join(str(invariant) for invariant in self.branches[i].invariants))
                ret += '\n'
                ret += indent(self.branches[i], depth+1)
                ret += '\n'
                ret += '\t'*depth
                ret += '}'
            elif self.branches[i].condition == None:
                ret += ' else {\t'
                ret += '‹{0}› '.format(' & '.join(str(constraint) for constraint in self.branches[i].constraints))
                ret += '«{0}»'.format(' & '.join(str(invariant) for invariant in self.branches[i].invariants))
                ret += '\n'
                ret += indent(self.branches[i], depth+1)
                ret += '\n'
                ret += '\t'*depth
                ret += '}'
            else:
                ret += ' elif '
                ret += indent(self.branches[i].condition, 0)
                ret += ' {\n'
                ret += '‹{0}› '.format(' & '.join(str(constraint) for constraint in self.branches[i].constraints))
                ret += '«{0}»'.format(' & '.join(str(invariant) for invariant in self.branches[i].invariants))
                ret += '\n'
                ret += indent(self.branches[i], depth+1)
                ret += '\n'
                ret += '\t'*depth
                ret += '}'

        return ret

    def __repr__(self) -> Text:
        return indent(self)

    def propagate(self, additions: Set[Expression]=set()) -> None:
        # Down-propagate invariants
        for branch in self._branches:
            branch.propagate(additions)

    @property
    def branches(self) -> Tuple[Branch]:
        return self._branches

    @property
    def conditions(self) -> Set[Expression]:
        return set(self._mapping.keys())

    @property
    def scope(self) -> Set[Variable]:
        return reduce(lambda s1, s2: s1 | s2,
                      map(lambda branch: branch.scope, self._branches))

    @property
    def constraints(self) -> Set[Expression]:
        return reduce(lambda xs, ys: xs | ys,
                      map(lambda branch: branch.constraints,
                          self.branches))

    @property
    def invariants(self) -> Set[Expression]:
        return reduce(lambda xs, ys: xs | ys,
                      map(lambda branch: branch.invariants,
                          self.branches))


class Loop(Indented, Invariant, Sequence['Statement']):
    def __init__(self,
                 branch: Branch) -> None:
        branch = copy.deepcopy(branch)

        self._branch = branch

    def __getitem__(self, i: int) -> 'Statement':
        return self._branch[i]

    def __len__(self) -> int:
        return len(self._branch)

    def __indent__(self, depth: int=0) -> Text:
        ret = '\t'*depth
        ret += 'while '
        ret += indent(self.condition)
        ret += ' {\t'
        ret += '‹{0}› '.format(' & '.join(str(constraint) for constraint in self.constraints))
        ret += '«{0}»\n'.format(' & '.join(str(invariant) for invariant in self.invariants))
        ret += indent(self.branch, depth+1)
        ret += '\n'
        ret += '\t'*depth
        ret += '}'

        return ret

    def __repr__(self) -> Text:
        return indent(self)

    def propagate(self, additions: Set[Expression]=set()) -> None:
        # Down-propagate invariants
        self._branch.propagate(additions)

    @property
    def branch(self) -> Block:
        return self._branch

    @property
    def condition(self) -> Expression:
        return self._branch.condition

    @property
    def scope(self) -> Set[Expression]:
        return self._branch.scope

    @property
    def constraints(self) -> Set[Expression]:
        return self._branch.constraints

    @property
    def invariants(self) -> Set[Expression]:
        return self._branch.invariants

"""
When constructing the Definition, go down the entire block
and recursively apply invariants

"""
class Definition(Indented, Invariant):
    def __init__(self,
                 name:        Text,
                 block:       Block,
                 constraints: Set[Expression]=set(),
                 invariants:  Set[Expression]=set()) -> None:
        name        = copy.deepcopy(name)
        block       = copy.deepcopy(block)
        constraints = copy.deepcopy(constraints)
        invariants  = copy.deepcopy(invariants)

        self._name        = name
        self._constraints = constraints
        self._invariants  = invariants
        self._block       = block
        self.propagate(self._invariants)

    def __len__(self) -> int:
        return len(self.blocks)

    def __indent__(self, depth: int=0) -> Text:
        ret = '\t'*depth
        ret += 'definition ' + self.name
        ret += ' {\t'
        ret += '‹{0}› '.format(' & '.join(str(constraint) for constraint in self.constraints))
        ret += '«{0}»\n'.format(' & '.join(str(invariant) for invariant in self.invariants))
        ret += indent(self.block, depth+1)
        ret += '\t'*depth
        ret += '}'

        return ret

    def __repr__(self) -> Text:
        return indent(self)

    def propagate(self, additions: Set[Expression]=set()) -> None:
        # Down-propagate invariants
        self._constraints |= additions
        self._invariants  |= additions

        self._block.propagate(self._invariants)

    @property
    def name(self) -> Text:
        return self._name

    @property
    def block(self) -> Tuple[Block]:
        return self._block

    @property
    def scope(self) -> Set[Variable]:
        return self._block.scope

    @property
    def constraints(self):
        return self._constraints

    @property
    def invariants(self):
        return self._invariants


class Trace(Sequence[Expression]):
    def __init__(self) -> None:
        self._trace = []

    def __getitem__(self, i: int) -> Expression:
        return self._trace[i]

    def __len__(self) -> int:
        return len(self._trace)

    def __iter__(self):
        return iter(self._trace)

    def __lshift__(self, step: Tuple[Expression, Dict[Variable, Any]]):
        self._trace.append(copy.deepcopy(step))

    def __repr__(self):
        width = 80
        ret = ''

        for i in range(len(self._trace)):
            assert isinstance(self._trace[i], tuple)
            assert len(self._trace[i]) == 2

            if i > 0:
                ret += ' '*((width//2)-1)
                ret += '╱╲\n'

                ret += ' '*((width//2)-1)
                ret += '╳╳\n'

                for j in range(2):
                    ret += ' '*((width//2)-1)
                    ret += '╳╳\n'

                ret += ' '*((width//2)-1)
                ret += '╲╱\n'

            ret += '╒'
            ret += '═'*5
            ret += '═'
            ret += '═'*(width-8)
            ret += '╕\n'

            source = str(self._trace[i][0]).strip().replace('\t', '    ')
            wrapped = []

            for line in source.splitlines():
                if len(line) > width-10:
                    wrapped.append(line[:width-11]+' \\')
                    wrapped.append(line[width-11:])
                else:
                    wrapped.append(line)

            header = width-2
            label  = str(type(self._trace[i][0]).__name__).upper()

            ret += '│'
            ret += ' '*((header-len(label)) // 2)
            ret += label
            ret += ' '*int(np.ceil((header-len(label)) / 2))
            ret += '│\n'
            ret += '╞'
            ret += '═'*5
            ret += '╦'
            ret += '═'*(width-8)
            ret += '╡\n'

            for i in range(len(wrapped)):
                ret += '│'
                ret += ' %03d ║ ' % (i+1,)
                ret += wrapped[i]
                ret += ' '*(width-len(wrapped[i])-9)
                ret += '│\n'

            ret += '╞'
            ret += '═'*5
            ret += '╩'
            ret += '═'*(width-8)
            ret += '╡\n'
            ret += '│'
            label = 'VARIABLES'
            ret += ' '*((header-len(label)) // 2)
            ret += label
            ret += ' '*int(np.ceil((header-len(label)) / 2))
            ret += '│\n'

            table = OrderedDict()
            left_width = len(max(self._trace[i][1].keys(), key=lambda k: len(k)))

            if left_width % 2 == 0:
                left_width += 1

            for variable, expr in self._trace[i][1].items():
                value = str(expr)

                margin  = width-left_width-7
                padding = int(np.ceil(len(value) / margin))

                # Make padding odd so we can center values
                if padding % 2 == 0:
                    padding += 1

                right_pad = int(np.ceil(len(value) / padding))

                table[variable] = {
                    'value':        value,
                    'padding':      padding,
                    'right_pad':    right_pad
                }

            c = 1
            ret += '├'
            ret += '─'*(left_width+2)
            ret += '┬'
            ret += '─'*(width-left_width-5)
            ret += '┤\n'

            for variable in table:
                value       = table[variable]['value']
                padding     = table[variable]['padding']
                midpoint    = int(np.ceil(padding/2))
                right_pad   = table[variable]['right_pad']
                right_width = abs(right_pad - (width-left_width-5))

                ret += '│ '

                if padding == 1:
                    ret += ' '*((left_width-len(variable))//2)
                    ret += variable
                    ret += ' '*int(np.ceil((left_width-len(variable))/2))
                else:
                    ret += ' '*left_width

                ret += ' │'

                slices = len(value[:right_pad])
                endl   = int(np.ceil(right_width / 2))

                for j in range(padding):
                    segment = value[(j*right_pad):((j+1)*right_pad)]

                    ret += ' '*(right_width // 2)
                    ret += segment
                    ret += ' '*(endl+slices-len(segment))

                    if j != padding-1:
                        ret += '│\n│ '

                        if j == midpoint-2:
                            ret += ' '*((left_width-len(variable))//2)
                            ret += variable
                            ret += ' '*int(np.ceil((left_width-len(variable))/2))
                        else:
                            ret += ' '*(left_width)

                        ret += ' │'

                ret += '│\n├' if c < len(table) else '│\n└'
                ret += '─'*(left_width+2)
                ret += '┼' if c < len(table) else '┴'
                ret += '─'*(width-left_width-5)
                ret += '┤\n' if c < len(table) else '┘\n'

                c += 1

        return ret


Statement = Union[Definition,
                  Loop,
                  Conditional,
                  Branch,
                  Block,
                  Assignment,
                  Expression]

class ConstraintViolation(Exception):
    pass

class InvariantViolation(ConstraintViolation):
    pass

# Our way of getting around the subclassing Callable issue
class EngineBase(Generic[T]):
    @abstractmethod
    def __call__(self, statement: T) -> None:
        pass

class Engine(EngineBase[Statement]):
    def __init__(self,
                 program:   Definition,
                 eps:       Real=1e-9,
                 max_it:    int=10) -> None:
        program = copy.deepcopy(program)
        eps     = copy.deepcopy(eps)
        max_it  = copy.deepcopy(max_it)

        self._program = program
        self._variables = {}

        for variable in program.scope:
            self._variables[variable] = Symbol(variable)

        self._trace = Trace()
        self.eps = eps
        self.max_it = max_it

        self._trace << (None, self.variables)

    def __call__(self, **kwargs) -> Dict[Variable, Any]:
        for variable in self._variables:
            self._variables[variable] = Symbol(variable)

        for key in kwargs:
            assert key in self._variables
            assert isinstance(kwargs[key], Real)

            self._variables[key] = kwargs[key]

        self._execute(self._program)

        return self._variables

    @property
    def steps(self) -> int:
        return len(self.self._trace)

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def program(self) -> Definition:
        return self._program

    @property
    def variables(self) -> Dict[Variable, Any]:
        return self._variables

    def _execute(self, statement: Statement) -> None:
        # Would like to preserve line numbers here
        if isinstance(statement, Constrained):
            for constraint in statement.constraints:
                if not self._legal(constraint):
                    raise ConstraintViolation(constraint, self.variables)

        if isinstance(statement, Definition):
            self._define(statement)
        elif isinstance(statement, Loop):
            self._loop(statement)
        elif isinstance(statement, Conditional):
            self._cond(statement)
        elif isinstance(statement, Branch):
            self._branch(statement)
        elif isinstance(statement, Block):
            self._block(statement)
        elif isinstance(statement, Assignment):
            self._assign(statement)
        elif isinstance(statement, Expression):
            self._expr(statement)
        else:
            print(statement, type(statement))
            raise TypeError('Statement provided is not valid')

        if isinstance(statement, Invariant):
            for invariant in statement.invariants:
                if not self._legal(invariant):
                    raise InvariantViolation(invariant, self.variables)

        self._trace << (statement, self.variables)

    def _define(self, define: Definition) -> None:
        self._execute(define.block)

    def _loop(self, loop: Loop) -> None:
        for i in range(self.max_it):
            if self._legal(loop.condition):
                self._execute(loop.branch)
            else:
                break

    def _cond(self, cond: Conditional) -> None:
        for branch in cond:
            if branch.condition is None or self._legal(branch.condition):
                self._execute(branch)
                break

    def _branch(self, branch: Branch) -> None:
        self._execute(branch.block)

    def _block(self, block: Block) -> None:
        for statement in block:
            self._execute(statement)

    def _assign(self, assign: Assignment) -> None:
        if assign.var not in self._variables:
            raise TypeError('Provided variable not in symbol table: {0}'.format(assign.var))

        self._variables[assign.var] = simplify(self._expr(assign.expr))

    def _legal(self, expr: Expression) -> bool:
        args = []

        for arg in expr.args:
            if arg in self._variables:
                args.append(self._variables[arg])
            elif isinstance(arg, Expression):
                args.append(self._expr(arg))
            elif isinstance(arg, Variable):
                raise TypeError('Provided variable not in symbol table: {0}'.format(arg))
            else:
                args.append(sympify(arg))

        res = rel('_legal', lambda e: e != 0, 1)(expr.expr(*args))
        # res = solveset(expr.expr(*args) != 0, domain=S.Reals)

        if isinstance(res, EmptySet):
            print('↯'.format(args))

        return not isinstance(res, EmptySet)

    def _expr(self, expr: Expression) -> Any:
        args = []

        for arg in expr.args:
            if arg in self._variables:
                args.append(self._variables[arg])
            elif isinstance(arg, Expression):
                args.append(self._expr(arg))
            elif isinstance(arg, Variable):
                raise TypeError('Provided variable not in symbol table: {0}'.format(arg))
            else:
                args.append(sympify(arg))

        return expr.expr(*args)


# class f(Function)
class Relation(Function):
    @classmethod
    def eval(cls, *args):
        assert len(args) in cls.nargs

        depen = args[:]
        indep = None

        while len(depen) > 0:
            search = []

            for var in depen:
                if isinstance(var, Symbol):
                    indep = var
                    search = []
                    break
                elif isinstance(var, Real):
                    continue
                else:
                    search.extend(var.args)

            depen = search

        if indep is not None:
            res = solveset(cls._relation(*args), indep, domain=S.Reals)
        else:
            res = solveset(cls._relation(*args), domain=S.Reals)

        if isinstance(res, EmptySet):
            return S.Zero
        else:
            return S.One

def rel(name: Text, relation: Callable[..., bool], nargs: int):
    return type(name, (Relation,),
                {'nargs': nargs,
                 '_relation': relation})

# Can deal with memory by adding a stochastic term with a sizeof variable
# To determine how many of the stochastic terms are taken
# This would be a huge pain in the ass to program
def main(ac: int, av: List[str]) -> None:
    from sympy import cos
    expr1 = Expression(cos, (Expression(cos, (Expression(sin, (Expression(cos, ('x',)),)),)),))
    expr2 = Assignment('x', Expression(sin, ('x',)))
    expr3 = Expression(sin, ('x',))
    expr4 = Assignment('z', Expression(cos, ('z',)))

    cnd1 = Expression(rel('LT', lambda x, y: x < y, 2), ('x', 'z')) # x < y
    cnd2 = Expression(rel('LT+5', lambda x, y: x+5 > y, 2), ('x', 'y')) # x < y

    l1 = Loop(Branch(
        cnd2,
        Block((
            expr2,
        )), invariants={expr3}
    ))

    c1 = Conditional((
        Branch(expr1,
               Block((
                   expr2,
               ))),
        Branch(None,
               Block((
                   expr2,
                   expr4
               )))
    ))

    prog = Definition('prog', Block((
        l1,
        c1,
        expr2
    )), invariants={cnd1})

    stuff = {}#{'x': .5, 'y': 1.0, 'z': 5.0}
    engine = Engine(prog)
    engine(**stuff)
    print(engine.variables)
    # print(engine.trace)

if __name__ == '__main__':
    try:
        main(len(sys.argv), sys.argv)
    except (KeyboardInterrupt, SystemError):
        pass