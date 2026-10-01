"""Build the GitHub Pages research paper from committed aggregate receipts."""
from pathlib import Path
from html import escape
import json
import re
import shutil

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "work/outputs"
DOCS = REPO / "docs"
GITHUB = "https://github.com/MTahaR-dev/flyrank-ml-internship"
SOURCE = GITHUB + "/blob/main/"
PAGE_URL = "https://mtahar-dev.github.io/flyrank-ml-internship/"


def load(filename):
    return json.loads((OUTPUTS / filename).read_text(encoding="utf-8"))


def recommendation_card(row):
    evidence = row["why_here"].replace("(-100.0%)", "(-99.97%, rounded)")
    return f'''<article class="recommendation">
      <div class="rec-rank" aria-hidden="true">{row['rank']:02d}</div>
      <div><h3>{escape(row['recommendation'])}<span class="client-tag">{escape(row['client_alias'])}</span></h3>
      <p>{escape(evidence)}</p>
      <p class="rec-action">Action: investigate; refresh only if content review supports it.</p>
      <p class="rec-caveat"><strong>What could make this wrong:</strong> {escape(row['what_would_make_it_wrong'])}.</p>
      <span class="reason-code">Reason: {escape(row['reason_code'])} · Directional; benefit unverified</span></div>
    </article>'''


def build_paper():
    feature = load("capstone_feature_table.json")
    metrics = load("capstone_notebook_metrics.json")
    visuals = load("capstone_visuals.json")
    recommendations = load("capstone_recommendations.json")
    baseline = load("w04_baseline_metrics.json")
    assert feature["labeled_pages"] == metrics["training_pages"] + metrics["test_pages"]
    assert recommendations["past_only_pages"] == feature["past_only_pages"]
    assert metrics["test_population_sha256"] == baseline["test_population_sha256"]
    assert len(recommendations["recommendations"]) == 20

    model_result, rule_result = metrics["test_comparison"]
    model_p20 = model_result["precision_at_20_pct"]
    rule_p20 = rule_result["precision_at_20_pct"]
    random_p20 = visuals["expected_random_precision_at_20_pct"]
    cv = metrics["training_cross_validation"]
    topics = [("abstract", "Abstract"), ("problem", "Problem"), ("data", "Data"),
              ("method", "Methodology"), ("results", "Results"), ("limitations", "Limitations"),
              ("recommendations", "Recommendations"), ("reproducibility", "Reproducibility"),
              ("credits", "Data credit")]
    nav = "".join(f'<a href="#{key}"><span>{n:02d}</span>{label}</a>'
                  for n, (key, label) in enumerate(topics, 1))
    client_rows = "".join(
        f'''<tr data-client="{escape(row['client_alias'])}"><td>{escape(row['client_alias'])}</td>
        <td class="numeric">{row['pages']:,}</td><td class="numeric">{row['model_precision_at_20_pct']:.1f}%</td>
        <td class="numeric">{row['rule_precision_at_20_pct']:.1f}%</td>
        <td class="numeric">{row['random_expected_precision_at_20_pct']:.1f}%</td></tr>'''
        for row in visuals["test_client_results"])
    options = "".join(f'<option value="{escape(row["client_alias"])}">{escape(row["client_alias"])}</option>'
                      for row in visuals["test_client_results"])
    cards = [recommendation_card(row) for row in recommendations["recommendations"]]
    client_json = json.dumps(visuals["test_client_results"], ensure_ascii=False).replace("<", chr(92) + "u003c")
    notebook_link = SOURCE + "work/notebooks/capstone.ipynb"

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which pages deserve a closer look? | FlyRank capstone research</title>
<meta name="description" content="A reproducible study of 57,459 FlyRank pages: logistic regression versus a transparent rule for ranking content review candidates, with honest client-grouped validation.">
<meta name="theme-color" content="#fbfaf6"><meta property="og:title" content="Which pages deserve a closer look?">
<meta property="og:description" content="A simple rule narrowly led a five-feature model at top-20 decline-risk prioritization. Read the methods, results, and limitations.">
<meta property="og:type" content="article"><link rel="canonical" href="{PAGE_URL}">
<link rel="stylesheet" href="assets/paper.css"><script src="assets/paper.js" defer></script></head>
<body><a href="#paper" class="skip-link">Skip to the research paper</a>
<header class="topbar"><div class="topbar-inner"><a class="brand" href="#paper" aria-label="Search intelligence research paper">
<svg viewBox="0 0 30 30" aria-hidden="true"><rect x="2" y="17" width="6" height="11" fill="#4267b2"/><rect x="12" y="10" width="6" height="18" fill="#087f8c"/><rect x="22" y="2" width="6" height="26" fill="#192c38"/></svg>Search / notes</a>
<div class="header-actions"><a href="{GITHUB}">View repository ↗</a><button type="button" class="print-button" id="print-paper">Print paper</button></div></div></header>
<div class="layout"><aside class="sidebar"><div class="nav-label">In this paper</div><nav aria-label="Table of contents">{nav}</nav>
<p class="sidebar-note">Lane 2<br>Refresh / Content Opportunity Scoring<br><br>March 2026 evidence<br>October 2026 write-up</p></aside>
<main id="paper"><details class="mobile-nav"><summary>Explore the paper</summary><nav aria-label="Mobile table of contents">{nav}</nav></details>
<header class="hero"><p class="eyebrow">Search intelligence · Capstone research</p><h1>Which pages deserve<br>a closer look?</h1>
<p class="subtitle">Ranking content review candidates from past search performance—and testing whether a learned model improves on a simple rule.</p>
<div class="byline"><span>By <a href="https://github.com/MTahaR-dev">MTahaR-dev</a></span><span>Lane 2 · FlyRank ML Internship</span><span>October 2026</span></div>
<p class="hero-finding"><strong>The model did not beat the rule on our primary metric.</strong> In seven clients’ top-20 lists, logistic regression identified 85 later-declining pages; the rule identified 86. The result supports a cautious review queue, not automatic refresh decisions.</p></header>

<section id="abstract"><span class="section-number">01 / Title + Abstract</span><h2>A useful queue starts with an honest comparison</h2>
<p class="abstract-text">Can past search performance help an editor rank pages for review before their visibility declines? We analyzed {feature['labeled_pages']:,} labeled pages from {feature['clients']} pseudonymized clients in the March 2026 FlyRank warehouse release. Five features from March 1–14 fed a logistic regression model, which we compared with a frozen two-signal rule using separate client groups and a later March 18–31 decline target. On the matched evaluation, the model reached {model_p20:.1f}% mean client Precision@20, the rule {rule_p20:.1f}%, and random ranking had an expected precision of {random_p20:.1f}%. We retain the rule as a provisional review queue, with human checks, because this observed result identifies decline risk rather than proving that a refresh will help.</p>
<div class="stat-grid"><div class="stat"><strong>{feature['labeled_pages']:,}</strong><span>Pages with complete<br>past and outcome windows</span></div><div class="stat"><strong>{feature['clients']}</strong><span>Pseudonymized clients<br>in the labeled analysis</span></div><div class="stat"><strong>{rule_p20:.1f}%</strong><span>Rule Precision@20<br>across seven test clients</span></div><div class="stat"><strong>1 page</strong><span>Observed rule advantage<br>in 140 recommendation slots</span></div></div></section>

<section id="problem"><span class="section-number">02 / Introduction · Problem statement</span><h2>Review capacity is scarce; a falling page is a clue</h2>
<p>The first page in our rule’s queue recorded 83,770 impressions in the past window, but its weekly count fell from 83,741 to 29. That deserves investigation. It does not, on its own, establish that the article is stale: tracking, indexation, demand, and intentional changes can produce a similar pattern.</p>
<p>The decision is <strong>which pages an editor should investigate first</strong>. We aim to provide a ranked queue with measured reasons, helping an editor decide whether to review content, investigate measurement, or monitor. A wrong recommendation wastes review time and may encourage unnecessary changes; a missed candidate can leave a problem unattended.</p>
<p>Data can make this triage repeatable. A learned model could combine exposure, clicks, position, and momentum in ways a two-signal rule misses, but complexity must earn its place on a metric tied to review capacity. Here, the model did not earn that place on the primary metric.</p>
<div class="callout"><span class="callout-label">Research question</span><p>Can past search-performance signals rank pages by later impression-decline risk better than a transparent rule, helping an editor choose which pages to review first?</p></div></section>

<section id="data"><span class="section-number">03 / Data</span><h2>One March slice, two non-overlapping windows</h2>
<dl class="definition-list"><dt>Source</dt><dd><a href="https://huggingface.co/datasets/FlyRank/internship-warehouse">FlyRank internship warehouse</a>, release v20260703, pinned revision <code>{feature['warehouse_revision']}</code>.</dd>
<dt>Table</dt><dd><code>fact_content_daily_performance</code>, March 2026 partition.</dd><dt>Raw grain</dt><dd>One report date × pseudonymized client × content item.</dd><dt>Prepared grain</dt><dd>One client/content item at the March 18, 2026 decision point.</dd></dl>
<div class="timeline" role="img" aria-label="March 1 to 14: features; March 15 to 17: excluded reporting buffer; March 18 to 31: outcome only">
<div class="timeline-part past-window"><strong>March 1–14</strong>Past inputs · 14 days</div><div class="timeline-part buffer-window"><strong>March 15–17</strong>Buffer · 3 days</div><div class="timeline-part future-window"><strong>March 18–31</strong>Outcome only · 14 days</div></div>
<p class="timeline-caption">Weeks 1 and 2 are March 1–7 and March 8–14. The buffer is an assumption because historical ingestion timestamps are unavailable.</p>
<p>A usable day requires <code>gsc_data_available IS TRUE</code>, observed impressions and clicks, and non-negative counts. Require all 14 distinct past dates and at least 100 past impressions: {feature['past_only_pages']:,} pages across 33 clients qualify for scoring. Requiring complete later coverage produces {feature['labeled_pages']:,} labeled pages across {feature['clients']} clients; {feature['pages_without_complete_target']:,} eligible past pages lack a complete target.</p>
<p>The labeled subset contains {feature['positive_targets']:,} declining pages ({100 * feature['positive_targets'] / feature['labeled_pages']:.2f}%) and {feature['negative_targets']:,} other pages. All five labeled features have zero missing values in this slice.</p>
<p>We exclude GA4 and AI-referral fields, text, client names, domains, page URLs, private queries, existing product scores, and incompatible 90-day query context. Availability flags prevent missing search history from becoming artificial zero activity. Pseudonymous identifiers are used for joins, grouping, and tie-breaking only. June is untouched; its final-month sample was not used to develop labels.</p></section>

<section id="method"><span class="section-number">04 / Methodology</span><h2>Learn from the past, evaluate on different clients</h2>
<h3>Five inputs available before the decision</h3><div class="table-wrap"><table class="feature-table"><thead><tr><th>Feature</th><th>Definition</th><th>Available because</th></tr></thead><tbody>
<tr><td>Search exposure</td><td><code>log(1 + past impressions)</code></td><td>Only March 1–14 counts are used.</td></tr><tr><td>Click volume</td><td><code>log(1 + past clicks)</code></td><td>Clicks precede the decision.</td></tr>
<tr><td>CTR</td><td><code>100 × clicks / impressions</code></td><td>The rate uses the same past window.</td></tr><tr><td>Average position</td><td>Impression-weighted valid daily positions</td><td>Only observed positive positions on past positive-impression days contribute.</td></tr>
<tr><td>Prior weekly change</td><td><code>100 × (week 2 − week 1) / week 1</code></td><td>Both weeks end before March 18.</td></tr></tbody></table></div>
<p class="aside-note">Position is an approximation from daily averages. A zero first-week denominator or unavailable position stays missing; any replacement is fitted on training clients only. Rate values are percentages.</p>
<h3>An observed decline target, not a refresh-success label</h3><p>The target is 1 when March 18–31 impressions are less than 80% of March 1–14 impressions, and 0 otherwise. Equal-length windows avoid unequal exposure duration. Later counts and later completeness never enter feature columns or recommendation scoring. Past impressions also form the label’s reference denominator, so this is a relative-change prediction problem.</p>
<div class="method-grid"><div><h3>Frozen rule</h3><p>Require at least 300 past impressions and a prior weekly fall of at least 20%. These are review-policy cutoffs, not universal SEO thresholds. Keep the Week 4 rule unchanged.</p></div><div><h3>Logistic regression</h3><p>Median imputation → standardization → logistic regression. Use L2 regularization, C=1, lbfgs, max_iter=1,000, and seed 42. No test-driven tuning.</p></div></div>
<code class="equation">rule score = log(1 + past impressions) × (−prior weekly change / 100)<br>when impressions ≥ 300 and prior change ≤ −20%; otherwise 0</code>
<h3>Client groups, training-only preparation, matched evaluation</h3><p><code>GroupShuffleSplit</code> with seed 42 assigns 24 clients ({metrics['training_pages']:,} pages) to training and eight clients ({metrics['test_pages']:,} pages) to testing. No client or page crosses the split. The 25% test fraction refers to clients; different inventories explain the unequal page counts.</p>
<p>Four-fold <code>GroupKFold</code> inside the training clients produces out-of-fold predictions. Each fold fits its own imputer, scaler, and classifier; the final model fits all training pages. Test outcomes were previously examined in weekly assignments, so this is a reused evaluation population rather than a blind sealed test. Features precede outcomes, but training and testing share the March cutoff: this tests cross-client generalization, not future-month generalization.</p>
<p>This is a retrospective policy comparison: the classifier is fitted after the training clients’ March outcomes are known. It does not simulate a model already trained on earlier months at the March 18 decision point. The past-only queue is a historical decision-support illustration.</p>
<p><strong>Primary metric: mean client Precision@20.</strong> Rank each eligible client’s pages, measure how many of its first 20 later declined, then average across clients. Seven test clients have at least 20 labeled pages; one is excluded from this metric. Both methods use identical pages and content-ID tie-breaking. ROC-AUC is secondary and includes all test pages.</p>
<ul class="checks"><li>Check all input dates precede the later outcome window.</li><li>Keep labels, outcome counts, product flags, and IDs outside model inputs.</li><li>Check zero client overlap, row alignment, and the exact baseline test-population fingerprint.</li><li>Fit preprocessing within training data; reproduce the frozen baseline score.</li><li>Separate complete-case evaluation from the past-only recommendation population.</li></ul></section>

<section id="results"><span class="section-number">05 / Results</span><h2>The model improved overall discrimination, not top-20 precision</h2>
<p>The rule narrowly leads on the metric we chose before training. Both methods exceed the matched random-ranking expectation in the overall average, but the one-page difference does not establish reliable superiority.</p>
<div class="table-wrap"><table class="result-table"><thead><tr><th>Method</th><th class="numeric">Mean client Precision@20</th><th class="numeric">ROC-AUC</th></tr></thead><tbody>
<tr><td>Logistic regression</td><td class="numeric">{model_p20:.3f}%</td><td class="numeric">{model_result['roc_auc']:.3f}</td></tr>
<tr class="preferred"><td>Frozen Week 4 rule <span class="small-tag">RETAINED</span></td><td class="numeric">{rule_p20:.3f}%</td><td class="numeric">{rule_result['roc_auc']:.3f}</td></tr>
<tr class="reference"><td>Expected random ranking</td><td class="numeric">{random_p20:.3f}%</td><td class="numeric">0.500</td></tr></tbody></table></div>
<p class="aside-note">The equal-client random reference uses the same seven eligible clients. The pooled decline rate among all test pages is {100 * baseline['test_page_base_rate']:.2f}%; it differs because larger clients contribute more pages. The random row is an expectation, not one simulated draw.</p>
<figure><img src="assets/figures/test_precision_at_20.svg" width="1530" height="952" alt="Model precision 60.7 percent; frozen rule 61.4 percent; expected random 39.8 percent, across seven test clients."><figcaption>The model identifies 85 later-declining pages in 140 top-20 slots; the rule identifies 86. These methods may select overlapping pages, and we have not established statistical superiority.</figcaption></figure>
<p>Internal training-client validation reaches {100 * cv['precision_at_20']:.3f}% Precision@20 against a {100 * cv['expected_random_precision_at_20']:.3f}% random expectation across 16 eligible training clients, with pooled out-of-fold ROC-AUC {cv['roc_auc']:.3f}. This is a different population from the final comparison; its AUC pools scores from multiple fitted folds.</p>
<h3>Client differences matter more than the headline gap</h3><p>The model leads for four clients and the rule for three. The rule falls below its client-specific random expectation in two clients. A slightly higher overall mean therefore does not justify a universal claim that the rule is best.</p>
<figure><img loading="lazy" src="assets/figures/precision_by_client.svg" width="1700" height="1071" alt="Paired model and rule Precision at 20 bars for seven anonymous clients, with page counts and random expectations."><figcaption>Every client contributes 20 ranked slots, even though their eligible inventories range from 43 to 15,502 pages. Client labels are anonymous and specific to this evaluation table.</figcaption></figure>
<div class="interactive-control"><label for="client-select">Explore a test client</label><select id="client-select"><option value="all">All seven clients</option>{options}</select></div><p id="client-takeaway" class="client-takeaway" aria-live="polite"></p>
<div class="table-wrap"><table id="client-results"><thead><tr><th>Client</th><th class="numeric">Pages</th><th class="numeric">Model P@20</th><th class="numeric">Rule P@20</th><th class="numeric">Random</th></tr></thead><tbody>{client_rows}</tbody></table></div>
<h3>What the model learned—and where it missed</h3><p>The largest absolute standardized weights are click volume (−0.744), exposure (+0.560), and prior weekly change (−0.551). High exposure with few clicks and an earlier falling trend are plausible review clues. Clicks, impressions, and CTR are related by construction; the conditional weights do not isolate causal effects, and the positive CTR coefficient does not mean that improving CTR causes decline.</p>
<figure><img loading="lazy" src="assets/figures/model_feature_weights.svg" width="1615" height="986" alt="Signed standardized coefficients: clicks minus 0.744, exposure plus 0.560, prior weekly change minus 0.551, CTR plus 0.181, position plus 0.079."><figcaption>Weights describe the fitted logistic regression model, not Google’s algorithm. They apply to standardized features and must be interpreted together.</figcaption></figure>
<p>Three high-scoring false positives in the model’s top-20 lists all had zero past clicks. None met the later decline target, despite scores above 0.82:</p>
<div class="table-wrap"><table class="error-table"><thead><tr><th>Example</th><th class="numeric">Past weekly change</th><th class="numeric">Model score</th><th>Why it is hard</th></tr></thead><tbody>
<tr><td>1</td><td class="numeric">−8.85%</td><td class="numeric">0.8516</td><td>A high-exposure/no-click pattern did not imply later decline.</td></tr><tr><td>2</td><td class="numeric">+32.61%</td><td class="numeric">0.8500</td><td>The score stayed high despite positive past momentum.</td></tr><tr><td>3</td><td class="numeric">−38.48%</td><td class="numeric">0.8243</td><td>An earlier fall did not necessarily continue into the later window.</td></tr></tbody></table></div><p class="aside-note">These are numeric record reviews, not article inspections. The scores are uncalibrated; actual causes remain unknown.</p></section>

<section id="limitations"><span class="section-number">06 / Limitations &amp; honest framing</span><h2>A review signal is not a diagnosis</h2>
<ol class="limitation-list"><li><strong>One date, not a future-period benchmark.</strong> Cross-client performance at the March cutoff does not establish reliability in a later month. Training uses March outcomes retrospectively; a model trained before March 18 has not been demonstrated. June remains untouched.</li>
<li><strong>A reused evaluation population.</strong> Test clients were absent from fitting, but their outcomes were examined in earlier assignments. We do not claim a blind holdout.</li>
<li><strong>Complete-case selection.</strong> Later coverage is required to measure the label, favoring better-tracked pages. The production-style queue applies only past eligibility.</li>
<li><strong>Few clients and heterogeneous performance.</strong> Seven clients contribute Precision@20. The one-slot gap has no demonstrated statistical significance; the rule is below random expectation for two clients.</li>
<li><strong>Reporting and measurement assumptions.</strong> Historical ingestion timestamps are unavailable. The three-day buffer does not prove every past measurement was present at the decision moment; daily position weighting is approximate.</li>
<li><strong>No causal refresh claim.</strong> We did not observe or randomize refresh interventions. Scores are uncalibrated, coefficients are conditional associations, and this study cannot prove editorial benefit or explain Google’s ranking algorithm.</li>
<li><strong>A concentrated global queue.</strong> One client supplies 16 of the first 20 candidates. A portfolio-wide queue does not provide a balanced allocation across clients.</li></ol></section>

<section id="recommendations"><span class="section-number">07 / Ranked recommendations</span><h2>Investigate first. Refresh only with a reason.</h2>
<p>The frozen rule produces {recommendations['queue_pages']:,} positive-score candidates across {recommendations['queue_clients']} clients from all {recommendations['past_only_pages']:,} eligible past pages. Its unitless score orders review priorities; it is not a probability or a forecast of recoverable traffic. The full queue is generated locally and stays out of Git.</p>
<ol class="playbook"><li><strong>Choose a client and its review budget.</strong> Use within-client ranks to avoid letting one large or abruptly falling client consume the global shortlist.</li><li><strong>Verify measurement, demand, and indexation.</strong> Check tracking continuity and search context, especially near-collapses and pages with no clicks.</li><li><strong>Inspect the content before editing.</strong> Refresh only if an authorized content review finds a specific mismatch, outdated information, or missing intent; otherwise investigate or monitor.</li><li><strong>Record the action and follow-up.</strong> Track later observations without attributing any change causally to a refresh.</li></ol>
<p class="aside-note">The public dataset uses pseudonymous IDs. An authorized operator would need their own mapping to identify actual pages. These anonymous previews reveal no client names, domains, URLs, or queries.</p>
<p class="queue-stats">20 reviewed candidates · 3 clients represented · 16 candidates from one client</p>
{''.join(cards[:5])}<details class="more-recommendations"><summary>Read recommendations 06–20 and their caveats</summary>{''.join(cards[5:])}</details>
<p class="aside-note">All evidence above is from March 1–14. No outcome label, later count, or future availability restriction was used to score or explain these candidates. Percentages and scores are rounded.</p></section>

<section id="reproducibility"><span class="section-number">08 / Reproducibility</span><h2>Follow the evidence back to the notebook</h2>
<p>The capstone notebook loads the real warehouse slice, verifies its inputs, constructs five features and a separate target, creates the client split, fits and validates the model, reproduces the baseline, and generates the queue and figures. The student’s training script was written and run incrementally with AI assistance. The notebook records the same logic and retains actual execution outputs.</p>
<div class="artifact-links"><a href="{notebook_link}">Executed capstone notebook<span>Data → model → comparison → recommendations</span></a><a href="{SOURCE}scripts/train_capstone_model.py">Training source<span>The student-authored Python script</span></a><a href="{SOURCE}work/outputs/capstone_notebook_metrics.json">Notebook metrics receipt<span>Split fingerprint, settings, results, versions</span></a><a href="{SOURCE}work/outputs/capstone_recommendations.json">Recommendation receipt<span>Anonymous reviews and action policy</span></a></div>
<h3>Rerun from a fresh clone</h3><pre><code>git clone {GITHUB}.git
cd flyrank-ml-internship
python -m venv .venv
# Activate .venv in your terminal, then:
python -m pip install -r work/requirements-capstone.txt
python scripts/execute_capstone.py
python scripts/build_capstone_paper.py</code></pre>
<p>Before execution, request access to the Hugging Face warehouse and configure a read token securely as <code>HF_TOKEN</code> in your environment or use an existing local Hugging Face login. Alternatively, place the March parquet at <code>data/warehouse/month=2026-03/data_0.parquet</code>. Never put credentials in cells or Git. The notebook pins the release revision and avoids a full-warehouse scan.</p>
<p>Analysis was executed with Python 3.12.14, scikit-learn 1.7.2, pandas 3.0.1, NumPy 2.3.5, and DuckDB 1.4.3; the <a href="{SOURCE}work/requirements-capstone.txt">analysis requirements</a> record dependencies. The student’s original run used Python 3.14.6 and scikit-learn 1.9.0. Both receipts agree on Precision@20 and reported AUC. The charts were verified with matplotlib 3.11.1. Split and classifier seed: 42; grouped-validation folds: 4.</p>
<p>The plain-Python notebook runner executes cells in one shared namespace and records stdout and dataframe outputs, without requiring a Jupyter server. The <a href="{SOURCE}scripts/build_capstone_figures.py">figure renderer</a> reads aggregate receipts; the <a href="{SOURCE}scripts/build_capstone_paper.py">paper builder</a> reuses them and the verified figures. No raw data export or model binary is needed in the public site.</p>
<details><summary>Earlier assignment notebooks and additional receipts</summary><ul class="checks">
<li><a href="{SOURCE}work/notebooks/w01_research_question.ipynb">Week 1: research question</a></li><li><a href="{SOURCE}work/notebooks/w02_ml_task_framing.ipynb">Week 2: ML task framing</a></li><li><a href="{SOURCE}work/notebooks/w03_data_contract.ipynb">Week 3: contract and leakage experiment</a></li><li><a href="{SOURCE}work/notebooks/w04_baseline_score.ipynb">Week 4: frozen baseline</a></li><li><a href="{SOURCE}work/outputs/capstone_split.json">Client split receipt</a> · <a href="{SOURCE}work/outputs/capstone_visuals.json">Anonymous chart data</a> · <a href="{SOURCE}work/outputs/capstone_model_metrics.json">Original student-run metrics</a></li></ul></details></section>

<section id="credits"><span class="section-number">09 / Acknowledgments &amp; data credit</span><h2>Real search data, careful claims</h2>
<div class="credits"><p><strong>Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>.</strong> Thank you to FlyRank for the anonymized search warehouse and internship materials that made this research possible.</p><p>AI assistance supported step-by-step coding, notebook integration, verification, visualization, and preparation of this research page. Recommendations are decision-support hypotheses based on numeric evidence; no client-identifying content is published.</p></div></section>
<footer class="page-footer"><span>MTahaR-dev · Search intelligence capstone · October 2026</span><a href="#paper">Back to the question ↑</a></footer></main></div>
<script id="client-data" type="application/json">{client_json}</script></body></html>'''

    # Let mobile readers open the vector figures at full size.
    def wrap_figure(match):
        image = match.group(1)
        source = re.search(r'src="([^"]+)"', image).group(1)
        return (f'<figure><a href="{source}" target="_blank" rel="noopener" '
                f'aria-label="Open chart at full size">{image}</a>')
    page = re.sub(r'<figure>(<img [^>]+>)', wrap_figure, page)
    DOCS.mkdir(exist_ok=True)
    figure_folder = DOCS / "assets/figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    for name in ["test_precision_at_20", "precision_by_client", "model_feature_weights"]:
        shutil.copy2(OUTPUTS / f"figures/{name}.svg", figure_folder / f"{name}.svg")
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    print("Built docs/index.html from aggregate receipts; copied three public-safe SVG figures.")


if __name__ == "__main__":
    build_paper()
