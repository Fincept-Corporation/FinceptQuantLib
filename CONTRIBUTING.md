# Contributing to FinceptQuantLib

Thanks for helping improve FinceptQuantLib. Bug reports, fixes, new models and documentation
improvements are all welcome.

## Setup

```bash
git clone https://github.com/Fincept-Corporation/FinceptQuantLib.git
cd FinceptQuantLib
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest                 # unit tests and docstring examples
ruff check src tests   # lint
```

- Add tests for every bug fix and new feature. Numerical code should be checked against an
  independent reference (SciPy, a closed form, or a published value) in `tests/`.
- Keep docstring examples runnable; they are executed by `pytest`.
- Public functions and classes need type hints and a docstring describing arguments, units
  (decimal rates, year fractions) and return values.
- Keep the runtime dependencies to `scipy` and `python-dateutil`.
- Update `CHANGELOG.md` under an `Unreleased` heading.

## Reporting bugs

Open an issue with a minimal reproducible example, the expected result (with its source) and
the actual result, plus your Python and FinceptQuantLib versions.

## License

By contributing you agree that your contributions are licensed under the MIT License.
