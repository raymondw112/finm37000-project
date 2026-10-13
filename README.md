# finm37000-project

> **Status: Part 1 (planning).** This README describes the intended outcome and how to run the project. Sections marked **TODO** are to be agreed on by the team in the README PR. Run instructions are aspirational until the code lands.

## Team

| Member | GitHub | Part 1 role |
| --- | --- | --- |
| Ryota Hidaka | ryotahidaka | Tech leader (owns this repo) |
| Abinaya Ankur Raut | kira7688 | Communication leader (README) |
| Konark Gupta | konark20 | Design leader (Issues) |
| Raymond Wang | raymondw112 | Design leader (Issues) |

Team communication channel: **WhatsApp**

## Project Summary

#### Q) State the question or application, the CME market(s), and the expected output.

A) We study pairs trading on CME precious metals futures using Databento futures data and analyze the trading feasibility with proper backtesting.**

### Motivation

#### Q) Why is this interesting? What market-structure feature or trading idea does it explore?

A) At the times of inflation and market turmoil, investor prefer safe assets in the form of precious metals. Gold and Silver have seen a huge return in the recent past. However, when positioning in the precious metals, one has to understand the volatility of the asset and balance between these assets. We plan to analyse and come up with a strategy that tries to lever or de-lever out of these positions based on market movements.

### Scope

- **Market(s):** Commodites Previous Metals Market
- **Date range:** Past 2 years
- **Databento dataset and schema:** `GLBX.MDP3` (CME Globex). Schema(s): prices and trades
- **Out of scope:** TODO. Note that a Brent/WTI mean-reverting strategy is not permitted for this course.

### Desired Outcome

By the end of the project, a user can run a single documented workflow that:

1. downloads (or loads cached) Databento CME data,
2. cleans and transforms it,
3. runs the analysis, simulation, or strategy, and
4. writes results (tables, figures, a summary) to an output folder.

#### Q) Describe the final deliverable and the headline results we expect to show.

A) Precious metal investment strategy with levering/de-levering out of positions to get maximum returns and low drawdown.


## Course Logistics

### Data

Data comes from [Databento](https://databento.com/) under the course's CME license.

- A Databento API key is required. Set it as an environment variable; never commit it.
  ```bash
  export DATABENTO_API_KEY="your-key-here"
  ```
- Raw downloads are cached under `data/` (git-ignored) so the same request isn't billed or downloaded twice.
- **Additional data sources:** TODO (none planned yet). If added, document how to obtain them and how to run without them.

### Getting Started

**Requirements:** Python 3.12, git, a Databento API key.

```bash
git clone https://github.com/ryotahidaka/finm37000-project.git
cd finm37000-project

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pinned versions; also installs src/gcsi in editable mode

cp .env.example .env              # then paste your key into .env (git-ignored)
pytest                            # import smoke test; should pass on a fresh clone
```

The key is read from the `DATABENTO_API_KEY` environment variable: export it as above, or put it in `.env` (loading `.env` with `python-dotenv` is part of Issue #4). The course helper `finm37000.db_env_util.get_databento_api_key()` reads `~/.databento_api_key` instead, if you already set that up for class.

`requirements.txt` pins every package, including the course `finm37000` package at a fixed commit. Direct dependencies are declared in `pyproject.toml`. Use `pip install -e ".[dev]"` instead if you only want the latest compatible versions.

### Running the project

```bash
# 1. Fetch and cache data
python -m gcsi.data --start YYYY-MM-DD --end YYYY-MM-DD   # TODO

# 2. Run the analysis / strategy
python -m gcsi.run                                        # TODO

# 3. Results are written to output/
```

### Running the tests

```bash
pytest            # skips tests marked `db` (they need Databento access)
pytest -m db      # run only the Databento tests
```

### Repository Layout

```
finm37000-project/
├── README.md
├── pyproject.toml          # package metadata + direct dependencies
├── requirements.txt        # pinned full environment
├── .env.example            # copy to .env and add your Databento key
├── docs/DESIGN.md          # pipeline, components, design decisions
├── src/gcsi/               # pipeline code: data, rolls, cointegration, ou, signals, backtest, metrics
├── notebooks/              # exploration and presentation only
├── tests/
├── data/                   # cached Databento data (git-ignored)
└── output/                 # generated tables and figures (git-ignored)
```

See [docs/DESIGN.md](docs/DESIGN.md) for how the components fit together.

### Roadmap

The work is tracked in the [GitHub Issues](../../issues). Each issue maps to a step in the workflow above.

| Milestone | Course due | Goal |
| --- | --- | --- |
| Part 1 | Lecture 2 | Repo, README, and Issues agreed by all members |
| Part 2 | Lecture 3 | TODO (e.g. data pipeline and exploration) |
| Part 3 | Lecture 4 | TODO (e.g. core analysis or strategy) |
| Completion | Lecture 5 | Polished, reproducible project |

### Contributing / Workflow

1. Fork the main repo (owned by the Tech leader) and clone your fork.
2. Create a branch per issue (`git checkout -b issue-<n>-short-name`).
3. Open a Pull Request to the main repo that references the issue (`Closes #<n>`).
4. At least one other team member reviews; the Tech leader merges.
5. Discuss scope changes in the relevant Issue, not offline.

### References

- [Databento documentation](https://databento.com/docs)
- TODO: papers or prior work relevant to the chosen topic
