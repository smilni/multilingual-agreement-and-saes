"""This module is an unmodified transcription of
``syntaxgym/prediction.py`` from
https://github.com/cpllab/syntaxgym-core (commit on ``master``, v0.8a1),
with the single change that ``METRICS`` is inlined here instead of being
imported from ``syntaxgym.utils``.

Upstream file:
    https://github.com/cpllab/syntaxgym-core/blob/master/syntaxgym/prediction.py
License:
    MIT (see https://github.com/cpllab/syntaxgym-core/blob/master/LICENSE).
"""

from __future__ import annotations

from typing import List as TList, Optional as TOptional, Union

import numpy as np
from pyparsing import (
    ParseException,
    ParserElement,
    Suppress,
    Word,
    alphanums,
    infixNotation,
    nums,
    oneOf,
    opAssoc,
    pyparsing_common,
)

ParserElement.enablePackrat()

# Relative and absolute tolerance thresholds for surprisal equality.
# Verbatim from upstream; ``=`` in a SyntaxGym formula means
# ``np.isclose(lhs, rhs, rtol=EQUALITY_RTOL, atol=EQUALITY_ATOL)``.
EQUALITY_RTOL = 1e-5
EQUALITY_ATOL = 1e-3

# Within-region aggregation metrics supported by SyntaxGym test suites.
# Inlined from ``syntaxgym.utils.METRICS`` (upstream) so we don't need to
# import the ``docker``-dependent ``syntaxgym.utils``.
METRICS = {
    "sum":    sum,
    "mean":   np.mean,
    "median": np.median,
    "range":  np.ptp,
    "max":    max,
    "min":    min,
}


# ---------------------------------------------------------------------------
# Grammar for prediction formulae.
# ---------------------------------------------------------------------------
lpar = Suppress("(")
rpar = Suppress(")")
region = (
    lpar
    + (Word(nums) | "*")
    + Suppress(";%")
    + Word(alphanums + "_-")
    + Suppress("%")
    + rpar
)
literal_float = pyparsing_common.number


class Region:
    def __init__(self, tokens):
        self.region_number = tokens[0]
        self.condition_name = tokens[1]

    def __str__(self):
        return "(%s;%%%s%%)" % (self.region_number, self.condition_name)

    def __repr__(self):
        return "Region(%s,%s)" % (self.condition_name, self.region_number)

    def __call__(self, surprisal_dict):
        if self.region_number == "*":
            return sum(
                value
                for (condition, _region), value in surprisal_dict.items()
                if condition == self.condition_name
            )
        return surprisal_dict[self.condition_name, int(self.region_number)]


class LiteralFloat:
    def __init__(self, tokens):
        self.value = float(tokens[0])

    def __str__(self):
        return "%f" % (self.value,)

    def __repr__(self):
        return "LiteralFloat(%f)" % (self.value,)

    def __call__(self, surprisal_dict):
        return self.value


class BinaryOp:
    operators: TOptional[TList[str]] = None

    def __init__(self, tokens):
        self.operator = tokens[0][1]
        if self.operators is not None and self.operator not in self.operators:
            raise ValueError(
                "Invalid %s operator %s"
                % (self.__class__.__name__, self.operator)
            )
        self.operands = [tokens[0][0], tokens[0][2]]

    def __str__(self):
        return "(%s %s %s)" % (self.operands[0], self.operator, self.operands[1])

    def __repr__(self):
        return "%s(%s)(%s)" % (
            self.__class__.__name__,
            self.operator,
            ",".join(map(repr, self.operands)),
        )

    def __call__(self, surprisal_dict):
        op_vals = [op(surprisal_dict) for op in self.operands]
        return self._evaluate(op_vals, surprisal_dict)

    def _evaluate(self, evaluated_operands, surprisal_dict):
        raise NotImplementedError()


class BoolOp(BinaryOp):
    operators = ["&", "|"]

    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "&":
            return op_vals[0] and op_vals[1]
        if self.operator == "|":
            return op_vals[0] or op_vals[1]
        raise AssertionError(self.operator)


class FloatOp(BinaryOp):
    operators = ["-", "+"]

    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "-":
            return op_vals[0] - op_vals[1]
        if self.operator == "+":
            return op_vals[0] + op_vals[1]
        raise AssertionError(self.operator)


class ComparatorOp(BinaryOp):
    operators = ["<", ">", "="]

    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "<":
            return op_vals[0] < op_vals[1]
        if self.operator == ">":
            return op_vals[0] > op_vals[1]
        if self.operator == "=":
            return bool(
                np.isclose(
                    op_vals[0], op_vals[1],
                    rtol=EQUALITY_RTOL, atol=EQUALITY_ATOL,
                )
            )
        raise AssertionError(self.operator)


def Chain(op_cls, left_assoc: bool = True):
    def chainer(tokens):
        operators = tokens[0][1::2]
        args = tokens[0][0::2]
        if not left_assoc:
            raise NotImplementedError
        arg1 = args.pop(0)
        while args:
            operator = operators.pop(0)
            arg2 = args.pop(0)
            arg1 = op_cls([[arg1, operator, arg2]])
        return arg1

    return chainer


atom = region.setParseAction(Region) | literal_float.setParseAction(LiteralFloat)

prediction_expr = infixNotation(
    atom,
    [
        (oneOf("- +"), 2, opAssoc.LEFT, Chain(FloatOp)),
        (oneOf("< > ="), 2, opAssoc.LEFT, ComparatorOp),
        (oneOf("& |"), 2, opAssoc.LEFT, Chain(BoolOp)),
    ],
    lpar=lpar,
    rpar=rpar,
)


class Prediction:
    """SyntaxGym prediction formula evaluator (vendored upstream class).

    Parse a formula string once, then evaluate it against an item dict whose
    ``conditions`` list carries per-region ``metric_value[metric]`` surprisals.
    See ``src/evaluation/syntaxgym.py::evaluate_suite`` for how items are
    built from per-region log-probs computed by this repo's LM forward pass.
    """

    def __init__(
        self,
        idx: int,
        formula: Union[str, BinaryOp],
        metric: str,
    ):
        if isinstance(formula, str):
            formula_src = formula.replace("[", "(").replace("]", ")")
            try:
                formula = prediction_expr.parseString(formula_src, parseAll=True)[0]
            except ParseException as exc:
                raise ValueError(
                    "Invalid formula expression %r" % (formula,)
                ) from exc

        self.idx = idx
        self.formula = formula

        if metric not in METRICS:
            raise ValueError(
                "Unknown metric %s. Supported metrics: %s"
                % (metric, " ".join(METRICS.keys()))
            )
        self.metric = metric

    def __call__(self, item):
        surps = {
            (c["condition_name"], r["region_number"]): r["metric_value"][self.metric]
            for c in item["conditions"]
            for r in c["regions"]
        }
        return self.formula(surps)

    @property
    def referenced_regions(self):
        """Set of ``(condition_name, region_number)`` referenced by this formula.

        Wildcard references ``(*;%cond%)`` are collected into
        ``wildcard_conditions`` instead, because they expand to "sum of
        surprisals across *all* regions for ``cond``" at evaluation time and
        therefore cannot be represented by a single integer region number.
        """
        acc: set = set()
        wildcards: set = set()

        def traverse(x):
            if isinstance(x, BinaryOp):
                for val in x.operands:
                    traverse(val)
            elif isinstance(x, Region):
                if x.region_number == "*":
                    wildcards.add(x.condition_name)
                else:
                    acc.add((x.condition_name, int(x.region_number)))

        traverse(self.formula)
        self._wildcard_conditions = wildcards
        return acc

    @property
    def wildcard_conditions(self) -> set:
        """Conditions referenced via ``(*;%cond%)`` wildcard, if any."""
        if not hasattr(self, "_wildcard_conditions"):
            _ = self.referenced_regions
        return getattr(self, "_wildcard_conditions", set())
