# IRT Models

semla fits Item Response Theory models using CFA parameterization. This means IRT models benefit from the same estimation infrastructure (ML, diagnostics, fit indices) as CFA/SEM models.

## Supported Models

| Model | Items | Description |
|-------|-------|-------------|
| `1PL` | Binary | Rasch model -- equal discrimination, estimate difficulty |
| `2PL` | Binary | Two-parameter -- estimate both discrimination and difficulty |
| `GRM` | Ordinal | Graded Response Model -- for Likert-scale items |

## Basic Usage

```python
from semla import irt

# 2PL model for binary items
fit = irt(
    items=["item1", "item2", "item3", "item4", "item5"],
    data=df,
    model_type="2PL",
)

fit.summary()
```

## IRT Parameters

```python
fit.irt_params()
```

Returns a DataFrame with:
- **discrimination** (a) -- how well the item differentiates between ability levels
- **difficulty** (b) -- the ability level where P(correct) = 0.5
- Standard errors for each parameter

## Item Characteristic Curves

```python
icc = fit.icc()
```

Returns P(correct | theta) for each item across a range of ability values. Useful for visualizing how items perform across the ability continuum.

```python
import matplotlib.pyplot as plt

icc = fit.icc()
for col in icc.columns[1:]:  # skip theta column
    plt.plot(icc["theta"], icc[col], label=col)
plt.xlabel("Ability (theta)")
plt.ylabel("P(correct)")
plt.legend()
plt.show()
```

## Item and Test Information

```python
# Per-item information
fit.item_information()

# Total test information and SE of ability
test_info = fit.test_information()
# Columns: theta, information, se
```

High information = precise measurement at that ability level. The SE column is 1/sqrt(information).

## Ability Estimation

```python
# Regression-based ability scores (default)
abilities = fit.abilities(method="regression")

# Bartlett scores
abilities = fit.abilities(method="bartlett")
```

Returns a DataFrame with an `ability` column (theta estimates for each person).

## Graded Response Model

For ordinal items (e.g., Likert scales):

```python
fit = irt(
    items=["q1", "q2", "q3", "q4"],
    data=df,
    model_type="GRM",
)

fit.irt_params()   # includes threshold parameters for each category
fit.icc()          # category response curves
```

## Rasch Model (1PL)

The 1PL constrains all discriminations to be equal:

```python
fit = irt(items=["item1", "item2", "item3"], data=df, model_type="1PL")
```
