# ANALYTiC 3.0

An active learning system for trajectory classification, with an interface
redesigned around human–computer interaction principles.

ANALYTiC presents unlabeled movement trajectories on an interactive map, asks the
user to label the most informative ones (chosen by an active learning strategy),
and progressively trains a classifier on those labels. The result is a labeled
dataset plus visual model diagnostics — all from a browser UI backed by a small
Bottle server.

Tested with Python 3.11.5.

## Authorship

The original ANALYTiC system was designed and implemented by **Amílcar Soares**,
with Chiara Renso and Stan Matwin, and published in *IEEE Computer Graphics and
Applications* in 2017.

This version (3.0) was implemented by **Abd Al-Munaem Jalboot** as a Master's
thesis at Linnaeus University, Department of Computer Science and Media
Technology, supervised by Benjamin Powley and Amílcar Soares Junior, and examined
by Marcelo Milrad. It integrates HCI principles into the interface to enhance
usability while preserving the original system's active learning functionality.

## Citation

If you use this code in any circumstances, please cite the original paper:

> Júnior, Amílcar Soares, Chiara Renso, and Stan Matwin.
> "ANALYTiC: An active learning system for trajectory classification."
> *IEEE Computer Graphics and Applications* 37.5 (2017): 28-39.

- [Official link](https://ieeexplore.ieee.org/abstract/document/8047427)
- [ResearchGate link](https://www.researchgate.net/publication/319947527_ANALYTiC_An_Active_Learning_System_for_Trajectory_Classification)
- [Original repository](https://github.com/amilcarsj/analytic)

For the interface redesign in this version, please also cite:

> Jalboot, Abd Al-Munaem.
> "Integrating Human-Computer Interaction Principles into the Interface Design of
> ANALYTiC Platform to Enhance Usability."
> Master's thesis (30 HE credits), Linnaeus University, 2026.

- [DiVA record](https://lnu.diva-portal.org/smash/record.jsf?pid=diva2%3A2057376)
- [Full text (PDF)](https://lnu.diva-portal.org/smash/get/diva2:2057376/FULLTEXT01.pdf)

## What's new in 3.0

The redesign was driven by a heuristic evaluation of the original interface using
Nielsen's 10 Usability Heuristics and Shneiderman's Eight Golden Rules, which
surfaced seven usability problems. The resulting changes are visible throughout
the UI:

| Change | Where | Heuristic addressed |
| --- | --- | --- |
| Landing page explaining the system, its workflow, and its applications before the tool is entered | `home.html` | Help and documentation |
| Contextual info tooltips on every panel | `index.html` (`glyphicon-info-sign`) | Help and documentation |
| Snackbar that names the next action ("Click on the Run button to classify all trajectories") | `src/js/components/middle-nav-components.js:95` | Visibility of system status |
| Live labeling progress: trajectories labeled, remaining budget, trajectory count | `index.html` (`#total-labeled`, `#remaining-budget`) | Visibility of system status |
| User-configurable labeling budget and bag size | `index.html` (`#total-budget`, `#bag-size`) | User control and freedom |
| Confirmation dialog before adding class labels | `index.html` (`#customModal`) | Error prevention |
| Explicit Restart and Home controls | `index.html:60` | User control and freedom |
| Collapsible results sections | `index.html` (`#collapse-toggle-row1`…`row3`) | Aesthetic and minimalist design |
| Loading indicator during long operations | `index.html` (`#loading`) | Visibility of system status |
| Result export: print view and CSV download | `index.html:367` | User control and freedom |
| Added UMAP projection alongside PCA and t-SNE | `analytic/umaptemp.py`, `src/js/components/umap.js` | — |

In a comparative study with six participants, the redesigned interface
("Platform B") scored a mean SUS of **80.8** ("excellent"), against **65.4** for
the original ("Platform A"). The `Platform(B):` timing logs printed by
`/classify_all` (`server.py:163`) are instrumentation from that study.

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv analytic-venv
```

Windows (PowerShell):

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
analytic-venv\Scripts\activate
```

macOS / Linux:

```bash
source analytic-venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install umap-learn
```

`umap-learn` is required by `analytic/umaptemp.py` for the UMAP projection view.
It is not pinned in `requirements.txt`, so install it separately.

### 3. Get the datasets

Download from [this Box folder](https://lnu.box.com/s/zreowc8qngmrydgse03lkoaz2mxe8xqh)
and place the `.json` files in the `Data/` directory at the repository root.

The server reads datasets from `DataDir = './Data/'` (`server.py:40`), so a
dataset named `geolife` must exist as `Data/geolife.json`. Datasets bundled in
this repo: `geolife`, `fishingvessels`, `hurricanes`, `animals`.

## Running

```bash
python server.py
```

Then open <http://127.0.0.1:8081/>.

On subsequent runs you only need to reactivate the environment first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
analytic-venv\Scripts\activate
python server.py
```

## Rebuilding the client bundle

The browser loads a single minified bundle, `dist/js/analytic.min.js`. If you
change anything under `src/js/`, regenerate it:

```bash
python minify_all_js.py
```

The script concatenates every `.js` file under `src/js/` (recursively) and
minifies the result with `jsmin`.

## Workflow

1. Pick a dataset — the server returns a seed bag of 20 random trajectories
   (`BagSize` in `server.py:89`) as GeoJSON, rendered on a Leaflet map.
2. Set a total labeling budget and a bag size, then label the trajectories shown.
   At least one trajectory of each class must be labeled before a strategy can run.
3. Run an active learning round. The chosen strategy uses the current labels to
   select the next, most informative bag; its size comes from the `bag-size`
   control, passed to `/al_run` as `time_step`. The remaining budget updates
   after each round.
4. Repeat until the budget is spent, then classify all remaining trajectories.
5. Inspect results: PCA / t-SNE / UMAP projections, cross-validated F1 scores,
   and decision-boundary plots.

### Active learning strategies

`Random Sampling`, `Uncertain Sampling`, `Query-by-committee`
(see `analytic/trajectory_manager.py:81`).

### Classifiers

`Logistic Regression`, `Random Forest`, `KNN`, `Decision Tree`, `Ada Boost`,
`Gaussian Naive Bayes` (see `analytic/trajectory_manager.py:63`).

The results panel separately evaluates `Random Forest`, `Extra Trees`, and
`Gradient Boosting` via 5-fold stratified cross-validation on weighted F1.

## Outputs

Classifying a dataset writes two files to the repository root:

| File | Written by | Contents |
| --- | --- | --- |
| `<dataset>_label.json` | `/classify_all` | `{tid: label}` for every trajectory |
| `<dataset>.csv` | `/perform_dimensionality_reduction` | trajectory features joined with labels |

`/train_model`, `/predict`, and `/getPlots` all read `<dataset>.csv`, so the
dimensionality-reduction step must run before them. Decision-boundary plots are
saved to `images/test_1.png` … `images/test_3.png`.

## HTTP API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Main UI (`home.html`) |
| GET | `/index` | Alternate UI (`index.html`) |
| GET | `/css`, `/analytic_js` | Minified CSS / JS bundles |
| GET | `/data/geolife`, `/data/fishingvessels`, `/data/hurricanes`, `/data/animals` | Random bag of trajectories as GeoJSON |
| POST | `/al_run` | Run an AL round; returns the next bag to label |
| POST | `/classify_all` | Classify every trajectory; writes `<dataset>_label.json` |
| POST | `/trajectories_geojson` | Fetch specific trajectories by `tid` |
| GET | `/perform_dimensionality_reduction?key=<dataset>` | PCA + t-SNE coordinates; writes `<dataset>.csv` |
| GET | `/train_model?key=<dataset>` | Cross-validated F1 scores |
| GET | `/predict?key=<dataset>` | PCA-projected points and labels |
| GET | `/getPlots?key=<dataset>&colorScale=<json>` | Decision-boundary plot images |
| GET | `/barchart_data?dataset=<dataset>` | Raw dataset JSON for charts |
| POST | `/umap_plot` | UMAP projection (form field `dataset`) |
| GET | `/download/<filename>` | Download a file from the repository root |

CORS is open (`Access-Control-Allow-Origin: *`) and the server binds to
`127.0.0.1` — it is intended for local, single-user research use, not
deployment as-is.

## Layout

```
analytic/          Server-side package
  al_strategies.py       Active learning strategies (Random, Uncertainty, QBC)
  classify_all.py        Full-dataset classification
  trajectory_manager.py  Strategy/classifier factories, label helpers
  http_get_solr_data.py  Dataset reads, GeoJSON assembly
  umaptemp.py            UMAP projection
src/js/            Client source: map, UI components, Leaflet plugins
src/css/           Client stylesheets
dist/              Built bundles served to the browser
Data/              Datasets (*.json) — not committed
images/            Static images and generated plots
home.html          Main UI page
index.html         Alternate UI page
server.py          Bottle app and routes
minify_all_js.py   Bundles src/js -> dist/js/analytic.min.js
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE), inherited from the
[original ANALYTiC repository](https://github.com/amilcarsj/analytic).
