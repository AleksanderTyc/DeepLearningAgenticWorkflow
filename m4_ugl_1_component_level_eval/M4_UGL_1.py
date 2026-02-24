#!/usr/bin/env python
# coding: utf-8

# # M4 Agentic AI - Adding a component-level eval to the research workflow
# 
# ## 1. Introduction
# 
# In the previous graded lab (M3), you built a tool-using research agent that carried out a workflow of three steps:
# 
# 1. Search the web for information.  
# 2. Reflect on its output.  
# 3. Publish a clear HTML report.  
# 
# Now, in this ungraded lab, you are going to focus on evaluating **one component of that workflow**: the *research step*.  
# 
# Instead of generating essays and refining them, here you will design a **component-level evaluation** to check the quality of sources returned by the research step.  
# 
# The evaluation will compare the URLs retrieved by the agent against a **predefined list of preferred domains** (e.g., `arxiv.org`, `nature.com`, `nasa.gov`).  
# 
# This allows you to quantify whether the system is pulling information from trustworthy sources, using an **objective, per-example ground truth evaluation**.
# 
# 
# ### 1.1. Lab overview
# 
# In the video, Andrew showed a case where web search results were of **poor quality**, making it difficult to trust the information retrieved.  
# Building on that example, in this lab you will evaluate the reliability of sources by comparing them against a **predefined list of preferred domains**.
# 
# For this evaluation, we’ll focus on the topic *“recent developments in black hole science”*, one of the examples highlighted in the course.  
# The idea is to verify whether the web search tool is returning sources from preferred domains, and to quantify the ratio of preferred vs. total results.
# 
# This evaluation will be implemented as a single function that performs an **objective, per-example check**. It will:
# 
# * Parse the Tavily output (our web search tool).  
# * Identify which URLs belong to the list of **preferred domains**.  
# * Compute the ratio of preferred vs. total retrieved sources.  
# * Return both a boolean flag (**PASS/FAIL**) and a Markdown-formatted summary that can be embedded directly into reports.  
# 
# <img src="M4-UGL-1.png" width="70%">  
# 
# 
# ### 1.2. 🎯 Learning outcomes
# 
# You will learn how to:
# 
# * Write a function that can check the search results of a web search API for **preferred sources**.  
# * Create an evaluation to verify if your sources come from your **preferred domains**.  
# * Add a **component-level evaluation** to the web search function.  
# 

# ## 2. Setup: Import libraries and load environment
# 
# As in previous labs, you start by importing the required libraries and initializing your environment.

# In[1]:


# =========================
# Imports
# =========================

# --- Standard library 
from datetime import datetime
import json
import re

# --- Third-party ---
from aisuite import Client

# --- Local / project ---
import research_tools
import utils

client = Client()


# ## 3. Research Step – `find_references`
# 
# In the graded lab, the function you implemented both **searched the web and wrote a draft report** in one step.
# 
# Here, we split the web search functionality into a separate function called `find_references`. This allows you to evaluate the search results independently of the writing and reflection steps, which we will leave out from this lab since we are only focusing on the output of the web search step.
# 
# Notice two key differences from the graded lab implementation:
# 
# * This new function uses **AISuite**, which automatically manages the tool calls for you (instead of writing manual tool-calling code with the OpenAI SDK).  
# * The function also informs the LLM of the **current date**, which helps improve relevance for time-sensitive queries.  
# 
# The role of `find_references` is to **gather external information** from tools such as **Arxiv**, **Tavily**, and **Wikipedia**.  
# Because the quality of these results directly shapes the outputs of the graded lab, this is the stage where you can apply **evaluation methods** — for example, checking whether the returned URLs come from your list of **preferred domains**.  

# In[2]:


def find_references(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    """Perform a research task using external tools (arxiv, tavily, wikipedia)."""

    prompt = f"""
    You are a research function with access to:
    - arxiv_tool: academic papers
    - tavily_tool: general web search (return JSON when asked)
    - wikipedia_tool: encyclopedic summaries

    Task:
    {task}

    Today is {datetime.now().strftime('%Y-%m-%d')}.
    """.strip()

    messages = [{"role": "user", "content": prompt}]
    tools = [
        research_tools.arxiv_search_tool,
        research_tools.tavily_search_tool,
        research_tools.wikipedia_search_tool,
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_turns=5,
        )
        content = response.choices[0].message.content
        return (content, messages) if return_messages else content
    except Exception as e:
        return f"[Model Error: {e}]"


# Run the following cell to test the research function.  
# This task will retrieve two recent papers on developments in black hole science and display the results.

# In[3]:


research_task = "Find 2 recent papers about recent developments in black hole science"
research_result = find_references(research_task)

utils.print_html(
    research_result,
    title="Research Function Output"
)
"""
Research Function Output
I found two papers related to recent developments in black hole science:

1. **Accretion onto Supermassive Black Holes in Quasars: Learning from Optical/UV Observations**
   - **Authors**: Paola Marziani, Deborah Dultzin-Hacyan, Jack W. Sulentic
   - **Published**: June 28, 2006
   - **Summary**: This paper discusses the complexities of accretion processes in quasars and active galactic nuclei, particularly focusing on the connection between spectral properties and physical parameters. It emphasizes recent efforts in estimating black hole mass and Eddington ratio using optical and UV broad emission lines, despite uncertainties. The study sheds light on correlations such as the "eigenvector 1 parameter space" and the "Baldwin effect", providing insights into accretion properties, broad line region structure, and source evolution.
   - [Read more](http://arxiv.org/abs/astro-ph/0606678v1) | [PDF](https://arxiv.org/pdf/astro-ph/0606678v1)

2. **Disturbing the Black Hole**
   - **Author**: Jacob D. Bekenstein
   - **Published**: May 13, 1998
   - **Summary**: This paper explores the conjecture that the horizon area of a near equilibrium black hole is an adiabatic invariant. It presents examples, including a Schwarzschild black hole perturbed by scalar fields, a Kerr black hole under scalar radiation, and a Reissner–Nordström black hole absorbing a charge. These examples clarify some conditions for the conjecture's validity and motivate a black hole quantization scheme.
   - [Read more](http://arxiv.org/abs/gr-qc/9805045v1) | [PDF](https://arxiv.org/pdf/gr-qc/9805045v1)

It seems the search returned older papers, perhaps due to a mismatch in indexing or availability on the database, rather than the most current research as per your request. If you want, I can attempt another search or method.
"""

# ## 4. Evaluation Step – Preferred Domains
# 
# Not all sources retrieved by web search are equally reliable.  
# In this lab, we focus on **just one step from the previous graded lab** — the `find_references` research step — and show how to design a **component-level evaluation** that checks whether the returned domains belong to a predefined list of **preferred domains**.  
# 
# This is an example of an **objective evaluation with a clear per-example ground truth**.  
# As a reminder from the lecture, recall the two axes of evaluation: along these axes we are working in the **upper-left quadrant** — objective evaluations with explicitly defined ground truth applied at the level of each example.
# 
# <img src='M4-UGL-1-Evaluations.png' width='80%'>
# 
# 
# ### Why component-level evaluations?
# 
# As Andrew mentioned in the lecture:  
# 
# - If the problem lies in web search (usually the **first step** in a graded lab workflow), rerunning the *entire* pipeline (search → draft → reflect) every time can be **expensive** and noisy.  
# - Small improvements in web search quality may be hidden by randomness introduced by later components.  
# - By evaluating the web search *alone*, you get a **clearer signal** of whether that component is improving.  
# 
# Component-level evals are also efficient when multiple teams are working on different pieces of a system: each team can optimize its own component using a clear metric, without needing to run or wait for full end-to-end tests.  
# 
# ### How do we evaluate?
# 
# Our evaluation here is **objective**, and so can be evaluated using code. It has an example-specific ground truth - the list of preferred sources for this black hole query. To build the eval, you will:
# 
# 1. Extract the URLs returned by Tavily.  
# 2. Compare them against a predefined list of **preferred domains** (e.g., `arxiv.org`, `nature.com`, `nasa.gov`).  
# 3. Compute the **ratio of preferred vs. total results**.  
# 4. Return a **PASS/FAIL flag** along with a Markdown-formatted summary.  
# 
# This provides a reproducible, low-cost metric that tells us whether the research component — and only this step from the graded lab — is pulling from trusted sources.
# 
# 

# In[4]:


# list of preferred domains for Tavily results
TOP_DOMAINS = {
    # General reference / institutions / publishers
    "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
    "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

    # CS/AI venues & indexes
    "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

    # Other reputable outlets
    "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

    # Extra domains (case-specific additions)
    "pbs.org", "nova.edu", "nvcc.edu", "cccco.edu",

    # Well known programming sites
    "codecademy.com", "datacamp.com"
}

def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio=0.4):
    """
    Evaluate whether plain-text research results mostly come from preferred domains.

    Args:
        TOP_DOMAINS (set[str]): Set of preferred domains (e.g., 'arxiv.org', 'nature.com').
        raw (str): Plain text or Markdown containing URLs.
        min_ratio (float): Minimum preferred ratio required to pass (e.g., 0.4 = 40%).

    Returns:
        tuple[bool, str]: (flag, markdown_report)
            flag -> True if PASS, False if FAIL
            markdown_report -> Markdown-formatted summary of the evaluation
    """

    # Extract URLs from the text
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Preferred Domains
No URLs detected in the provided text. 
Please include links in your research results.
"""

    # Count preferred vs total
    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        domain = url.split("/")[2]
        preferred = any(td in domain for td in TOP_DOMAINS)
        if preferred:
            preferred_count += 1
        details.append(f"- {url} → {'✅ PREFERRED' if preferred else '❌ NOT PREFERRED'}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    # Markdown report
    report = f"""
### Evaluation — Tavily Preferred Domains
- Total results: {total}
- Preferred results: {preferred_count}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}

**Details:**
{chr(10).join(details)}
"""
    return flag, report


# <div style="border:1px solid #93c5fd; border-left:6px solid #3b82f6; background:#dbeafe; border-radius:6px; padding:12px 14px; color:#1e3a8a; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">  
# <strong>🔎 Why this is an objective evaluation:</strong><br><br>  
# Each URL retrieved from Tavily is compared against a predefined list of <em>preferred domains</em> (<code>TOP_DOMAINS</code>):<br>  
# • If the domain matches → ✅ PREFERRED<br>  
# • Otherwise → ❌ NOT PREFERRED<br><br>  
# This yields a clear PASS/FAIL signal depending on whether the ratio of preferred sources exceeds a given threshold.  
# Because the ground truth (preferred vs. not preferred) is explicitly defined for each example, the evaluation is both <strong>objective</strong> and <strong>reproducible</strong>.  
# </div>
# 

# Run the cell to display sample preferred domains, the research results, and the evaluation summary (PASS/FAIL with details).

# In[5]:


utils.print_html(json.dumps(list(TOP_DOMAINS)[:4], indent=2), title="Sample Trusted Domains")

utils.print_html("<h3>Research Results</h3>" + research_result, title="Research Results")

flag, report = evaluate_tavily_results(TOP_DOMAINS, research_result)
utils.print_html("<pre>" + report + "</pre>", title="<h3>Evaluation Summary</h3>")
"""
Sample Trusted Domains
[
  "europa.eu",
  "noaa.gov",
  "datacamp.com",
  "mit.edu"
]

Research Results
<h3>Research Results</h3>I found two papers related to recent developments in black hole science:

1. **Accretion onto Supermassive Black Holes in Quasars: Learning from Optical/UV Observations**
   - **Authors**: Paola Marziani, Deborah Dultzin-Hacyan, Jack W. Sulentic
   - **Published**: June 28, 2006
   - **Summary**: This paper discusses the complexities of accretion processes in quasars and active galactic nuclei, particularly focusing on the connection between spectral properties and physical parameters. It emphasizes recent efforts in estimating black hole mass and Eddington ratio using optical and UV broad emission lines, despite uncertainties. The study sheds light on correlations such as the "eigenvector 1 parameter space" and the "Baldwin effect", providing insights into accretion properties, broad line region structure, and source evolution.
   - [Read more](http://arxiv.org/abs/astro-ph/0606678v1) | [PDF](https://arxiv.org/pdf/astro-ph/0606678v1)

2. **Disturbing the Black Hole**
   - **Author**: Jacob D. Bekenstein
   - **Published**: May 13, 1998
   - **Summary**: This paper explores the conjecture that the horizon area of a near equilibrium black hole is an adiabatic invariant. It presents examples, including a Schwarzschild black hole perturbed by scalar fields, a Kerr black hole under scalar radiation, and a Reissner–Nordström black hole absorbing a charge. These examples clarify some conditions for the conjecture's validity and motivate a black hole quantization scheme.
   - [Read more](http://arxiv.org/abs/gr-qc/9805045v1) | [PDF](https://arxiv.org/pdf/gr-qc/9805045v1)

It seems the search returned older papers, perhaps due to a mismatch in indexing or availability on the database, rather than the most current research as per your request. If you want, I can attempt another search or method.

Evaluation Summary
<pre>
### Evaluation — Tavily Preferred Domains
- Total results: 4
- Preferred results: 4
- Ratio: 100.00%
- Threshold: 40%
- Status: ✅ PASS

**Details:**
- http://arxiv.org/abs/astro-ph/0606678v1 → ✅ PREFERRED
- https://arxiv.org/pdf/astro-ph/0606678v1 → ✅ PREFERRED
- http://arxiv.org/abs/gr-qc/9805045v1 → ✅ PREFERRED
- https://arxiv.org/pdf/gr-qc/9805045v1 → ✅ PREFERRED
</pre>
"""

# ## Try yourself!
# 
# Now it’s your turn.  
# In this section, you can experiment directly with the **research step** and its **evaluation**:  
# 
# * **Topic**: choose a different topic to research.  
# * **Preferred domains**: edit or expand the `TOP_DOMAINS` list.  
# * **Evaluation ratio**: adjust the `min_ratio` (e.g., 0.4 = at least 40% preferred sources).  
# 
# Re-run the cells below after making your edits to see how the evaluation changes.
# 

# In[6]:


# === 5.1. Try it yourself: topic, ratio & preferred domains ===
# Edit these parameters before running the cell

topic = "recent developments in black hole science"   # <- Change the topic here
min_ratio = 0.4                                       # <- Change threshold (0.0–1.0)
run_reflection = True                                 # <- Set False to skip Step 4

# Short list of preferred domains (edit or expand as needed)
TOP_DOMAINS = {
    "wikipedia.org", "nature.com", "science.org", "arxiv.org",
    "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu"
}

# Show a sample of preferred domains
import json
utils.print_html(
    json.dumps(sorted(list(TOP_DOMAINS)), indent=2),
    title="<h3>Sample Preferred Domains</h3>"
)

# 1) Research
research_task = f"Find 2–3 key papers and reliable overviews about {topic}."
research_output = find_references(research_task)
utils.print_html(research_output, title=f"<h3>Research Results on {topic}</h3>")

# 2) Evaluate sources (preferred domains ratio)
flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio=min_ratio)
utils.print_html("<pre>" + eval_md + "</pre>", title="<h3>Evaluation Summary</h3>")
"""
Sample Preferred Domains
[
  "arxiv.org",
  "harvard.edu",
  "mit.edu",
  "nasa.gov",
  "nature.com",
  "science.org",
  "stanford.edu",
  "wikipedia.org"
]

Research Results on recent developments in black hole science
Here's an overview of recent developments in black hole science, drawn from both academic papers and general overviews:

### Key Academic Papers
1. **Accretion onto Supermassive Black Holes in Quasars: Learning from Optical/UV Observations**  
   - **Authors**: Paola Marziani, Deborah Dultzin-Hacyan, Jack W. Sulentic  
   - **Summary**: This paper explores the complexities of accretion processes in quasars and the challenges in connecting observed spectral properties to physical parameters. It emphasizes the importance of accurate measurements of broad emission line properties for understanding quasars and proposes potential improvements in estimating black hole mass, accretion rate, and spin.
   - [Read more](http://arxiv.org/abs/astro-ph/0606678v1) | [PDF](https://arxiv.org/pdf/astro-ph/0606678v1)

2. **Disturbing the Black Hole**  
   - **Author**: Jacob D. Bekenstein  
   - **Summary**: Bekenstein investigates conditions supporting the adiabatic invariance of a near-equilibrium black hole's horizon area. The paper explores different black hole scenarios that provide evidence for a proposed quantization scheme of black holes.
   - [Read more](http://arxiv.org/abs/gr-qc/9805045v1) | [PDF](https://arxiv.org/pdf/gr-qc/9805045v1)

3. **Randall-Sundrum Gravitons and Black Holes at the LHC**  
   - **Author**: K. M. Black  
   - **Summary**: This paper discusses models predicting extra spatial dimensions that could reveal dramatic effects at the Large Hadron Collider (LHC), including the production and decay of mini-black holes and gravitons.
   - [Read more](http://arxiv.org/abs/0805.3007v2) | [PDF](https://arxiv.org/pdf/0805.3007v2)

### Recent News and Discoveries
- **Gravitational Waves and Dark Matter**: Recent findings suggest that gravitational waves from black holes could soon reveal where dark matter is hiding. [Read more on ScienceDaily](https://www.sciencedaily.com/news/space_time/black_holes/)
  
- **Rotating Black Holes**: New observations indicate that rotating black holes may drag spacetime around them, shedding light on fundamental questions in physics. [Read more on IAI TV](https://iai.tv/articles/new-black-hole-discovery-uncovers-our-failure-to-understand-reality-auid-3473)
  
- **First "Black Hole Triple" Discovery**: Physicists have discovered a system containing a central black hole consuming a small star, occurring every 6.5 days. [Read more on MIT News](https://news.mit.edu/2024/physicists-discover-first-black-hole-triple-1023)

### Encyclopedia Overview
- **Primordial Black Holes (PBHs)**: These are hypothetical black holes that formed soon after the Big Bang due to the gravitational collapse of dense pockets of matter. Proposed by Yakov Zeldovich and Igor Novikov in 1966, their existence remains hypothetical.
  - [Read more on Wikipedia](https://en.wikipedia.org/wiki/Primordial_black_hole)

These resources should provide a comprehensive view of the current state of black hole research and recent discoveries.

Evaluation Summary
<pre>
### Evaluation — Tavily Preferred Domains
- Total results: 10
- Preferred results: 8
- Ratio: 80.00%
- Threshold: 40%
- Status: ✅ PASS

**Details:**
- http://arxiv.org/abs/astro-ph/0606678v1 → ✅ PREFERRED
- https://arxiv.org/pdf/astro-ph/0606678v1 → ✅ PREFERRED
- http://arxiv.org/abs/gr-qc/9805045v1 → ✅ PREFERRED
- https://arxiv.org/pdf/gr-qc/9805045v1 → ✅ PREFERRED
- http://arxiv.org/abs/0805.3007v2 → ✅ PREFERRED
- https://arxiv.org/pdf/0805.3007v2 → ✅ PREFERRED
- https://www.sciencedaily.com/news/space_time/black_holes/ → ❌ NOT PREFERRED
- https://iai.tv/articles/new-black-hole-discovery-uncovers-our-failure-to-understand-reality-auid-3473 → ❌ NOT PREFERRED
- https://news.mit.edu/2024/physicists-discover-first-black-hole-triple-1023 → ✅ PREFERRED
- https://en.wikipedia.org/wiki/Primordial_black_hole → ✅ PREFERRED
</pre>
"""

# ## 5. Takeaways
# 
# * You just saw how to evaluate the performance of **one component**: the `find_references` research step.  
# * Your component-level evaluation checked whether the retrieved URLs were in a predefined list of **preferred domains**.  
# * This is an example of an **objective evaluation** with a clear **per-example ground truth**.  
# * To build an evaluation set, you could design ~10 prompts covering different topics (astronomy, robotics, finance, etc.) and define preferred domains for each.  
# * The percentage of retrieved sources that matched the list of preferred domains provides a useful **metric** to guide improvements, such as adjusting the prompt or tool parameters.  
# * This approach is **simpler and cheaper** than evaluating full essays with reflection and rewrites, since you only focus on the web search component.  
# 
# <div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
# 
# 🎉 **Congratulations!**  
# 
# You designed a **component-level evaluation** that makes your research agent more reliable.  
# By directly checking the quality of sources, you introduced a safeguard that is **objective, reproducible, and cost-effective**.  
# 
# This aligns with the idea highlighted in Andrew’s lecture: *component-level evaluations* let you test individual pieces of an AI system without the overhead of evaluating the entire pipeline.  
# 
# </div>
# 
# 
# 

# 

# In[7]:


# === 5.1. Try it yourself: topic, ratio & preferred domains ===
# Edit these parameters before running the cell

topic = "scientific developments in psychology of stock exchange investing"   # <- Change the topic here
min_ratio = 0.4                                       # <- Change threshold (0.0–1.0)
run_reflection = True                                 # <- Set False to skip Step 4

# Short list of preferred domains (edit or expand as needed)
TOP_DOMAINS = {
    "wikipedia.org", "nature.com", "science.org", "arxiv.org",
    "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu"
}

# Show a sample of preferred domains
import json
utils.print_html(
    json.dumps(sorted(list(TOP_DOMAINS)), indent=2),
    title="<h3>Sample Preferred Domains</h3>"
)

# 1) Research
research_task = f"Find 2–3 key papers and reliable overviews about {topic}."
research_output = find_references(research_task)
utils.print_html(research_output, title=f"<h3>Research Results on {topic}</h3>")

# 2) Evaluate sources (preferred domains ratio)
flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio=min_ratio)
utils.print_html("<pre>" + eval_md + "</pre>", title="<h3>Evaluation Summary</h3>")
"""
Sample Preferred Domains
[
  "arxiv.org",
  "harvard.edu",
  "mit.edu",
  "nasa.gov",
  "nature.com",
  "science.org",
  "stanford.edu",
  "wikipedia.org"
]

Research Results on scientific developments in psychology of stock exchange investing
### Key Academic Papers on Psychology of Stock Exchange Investing

1. **[Efficiency of the Moscow Stock Exchange before 2022](https://arxiv.org/pdf/2207.10476v2)**
   - **Authors:** Andrey Shternshis, Piero Mazzarisi, Stefano Marmi
   - **Published:** 2022-07-21
   - **Summary:** This paper explores the efficiency of the Moscow Stock Exchange, using methods like entropy and Monte Carlo simulations to analyze stock price regularities and co-movements. It suggests that market inefficiencies could signal opportunities for profitable strategies.

2. **[E2EAI: End-to-End Deep Learning Framework for Active Investing](https://arxiv.org/pdf/2305.16364v1)**
   - **Authors:** Zikai Wei, Bo Dai, Dahua Lin
   - **Published:** 2023-05-25
   - **Summary:** This study presents an end-to-end deep learning framework for active investing, which involves factor selection, stock selection, and portfolio construction. The framework demonstrates effectiveness in market data experiments.

3. **[Kurt Lewin, Psychological Constructs, and Sources of Brain Cognitive Activity](https://arxiv.org/pdf/1711.01767v1)**
   - **Author:** Włodzisław Duch
   - **Published:** 2017-11-06
   - **Summary:** The paper suggests connecting cognitive and social psychology constructs to brain processes, aiming to better understand mind-brain-environment relationships, which are key to comprehending investment decisions.

### Overview from Wikipedia

- **Stock Market Overview**
  - The stock market consists of stock exchanges where ownership claims on businesses are traded. As of 2023, the total market capitalization of publicly traded stocks was $111 trillion worldwide. Investments are typically made following strategic guidelines. [(Read more on Wikipedia)](https://en.wikipedia.org/wiki/Stock_market)

### Insights from General Web Sources

1. **The Psychology of Stock Market Investing: Mastering Emotions and Behavioral Biases to Build Wealth**
   - **Author:** Hyun Kim
   - **Published on Amazon**
   - This book digs into investors’ common mistakes and methods to avoid them, emphasizing mastering emotions for better investing outcomes. [(Available on Amazon)](https://www.amazon.com/Psychology-Stock-Market-Investing-Behavioral/dp/1968387072)

2. **The Psychology of Stock Investing: Emotions, Biases, and Better Decision-Making**
   - Explores how understanding cognitive patterns helps in countering emotional biases, improving investment outcomes through systematic approaches and analytics. [(Read more)](https://www.findex.se/knowledge-base/portfolio-management/psychology-of-stock-investing)

3. **The Psychology Of Investing: How to Avoid Losing**
   - Discusses how emotions shape market trends, with regulatory insights into meme stock behaviors, highlighting common cognitive biases affecting investors. [(Read more on Forbes)](https://www.forbes.com/sites/jimosman/2023/04/30/the-psychology-of-investing-how-to-avoid-losing/) 

These resources provide a mix of theoretical, practical, and psychological insights into the complex nature of stock investing, emphasizing the importance of emotional intelligence and systematic strategies.

Evaluation Summary
<pre>
### Evaluation — Tavily Preferred Domains
- Total results: 7
- Preferred results: 4
- Ratio: 57.14%
- Threshold: 40%
- Status: ✅ PASS

**Details:**
- https://arxiv.org/pdf/2207.10476v2 → ✅ PREFERRED
- https://arxiv.org/pdf/2305.16364v1 → ✅ PREFERRED
- https://arxiv.org/pdf/1711.01767v1 → ✅ PREFERRED
- https://en.wikipedia.org/wiki/Stock_market → ✅ PREFERRED
- https://www.amazon.com/Psychology-Stock-Market-Investing-Behavioral/dp/1968387072 → ❌ NOT PREFERRED
- https://www.findex.se/knowledge-base/portfolio-management/psychology-of-stock-investing → ❌ NOT PREFERRED
- https://www.forbes.com/sites/jimosman/2023/04/30/the-psychology-of-investing-how-to-avoid-losing/ → ❌ NOT PREFERRED
</pre>
"""

# In[8]:


import os
from IPython.display import FileLink
os.listdir('.') # list current directory
"""
['M4-UGL-1-Evaluations.png',
 'M4-UGL-1.png',
 'M4_UGL_1.ipynb',
 'research_tools.py',
 'utils.py',
 '.ipynb_checkpoints',
 '__pycache__']
 """

# In[12]:


FileLink('utils.py') # provide a link to data file

