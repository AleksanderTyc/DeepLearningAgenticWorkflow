#!/usr/bin/env python
# coding: utf-8

# # M5 Agentic AI - Customer Service Agent
# 
# ## 1. Introduction
# 
# As Andrew explained in the lecture, *planning with code execution* means letting the LLM **write code that becomes the plan itself**.  
# Compared to plain-text or JSON-based plans, this approach is more expressive and flexible: the code not only documents the steps but can also execute them directly.
# 
# In this lab, you will implement this design pattern in practice.  
# Instead of asking the LLM to output a plan in JSON format and then manually executing each step, we will allow it to **write Python code** that directly captures multiple steps of a plan. By executing this code, we can carry out complex queries automatically.  
# 
# To make things concrete, we simulate a **sunglasses store** with an **inventory** of products and a set of **transactions** (sales, returns, balance updates). This example shows how the LLM can generate code to query or update records, demonstrating the flexibility of this pattern.
# 
# ### 1.1 Lab Overview
# We will:
# 1. Create simple **inventory** and **transaction** datasets.  
# 2. Build a **schema block** describing the data.  
# 3. Prompt the LLM to **write a plan as Python code** (with comments explaining each step).  
# 4. Execute the code in a sandbox to obtain the answer.  
# 
# ### 1.2 Learning Outcomes
# 
# By the end of this lab, you will be able to:
# 
# - **Explain** why letting the model write code (instead of JSON or plain text plans) enables richer, more flexible planning.  
# - **Prompt** an LLM to produce Python code with step-by-step comments that both documents and executes the plan.  
# - **Run** the generated code safely in a sandbox and interpret the results.  
# 
# This illustrates how *Code as Action* can outperform brittle tool chains and JSON-based planning approaches.

# ## 2. Setup

# In[1]:


# ==== Imports ====
from __future__ import annotations
import json
from dotenv import load_dotenv
from openai import OpenAI
import re, io, sys, traceback, json
from typing import Any, Dict, Optional
from tinydb import Query, where

# Utility modules
import utils      # helper functions for prompting/printing
import inv_utils  # functions for inventory, transactions, schema building, and TinyDB seeding

load_dotenv()
client = OpenAI()


# In the `inv_utils` module, we have functions like:
# 
# - `create_inventory()` – builds the sunglasses inventory.  
# - `create_transactions()` – builds the initial transaction log.  
# - `seed_db()` – loads both inventory and transactions into a JSON-backed store.  
# - `build_schema_block()` – generates a schema description used in the prompt.  
# - Helpers like `get_current_balance()` and `next_transaction_id()` – let the LLM handle consistent updates across inventory and transactions.  

# ### 2.1 Create Example Tables
# 
# We will now create two small tables for the sunglasses store simulation, using **[TinyDB](https://tinydb.readthedocs.io/)** — a lightweight document-oriented database written in pure Python.  
# TinyDB stores data as JSON documents and is well-suited for small applications or prototypes, since it requires no server setup and allows you to query and update data easily.
# 
# The two tables are:
# 
# - **`inventory_tbl`**: contains product details such as name, item ID, description, quantity in stock, and price.  
# - **`transactions_tbl`**: starts with an opening balance and will later track purchases, returns, and adjustments.  
# 
# You will generate these tables using helper functions in `inv_utils`, and then preview the first few rows below.

# In[2]:


db, inventory_tbl, transactions_tbl = inv_utils.seed_db()


# Now, you can inspect the records in each table by printing them as formatted JSON:

# In[3]:


utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table")
utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table")
"""
Inventory Table
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 23,
    "price": 80
  },
  {
    "item_id": "SG002",
    "name": "Wayfarer",
    "description": "Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.",
    "quantity_in_stock": 6,
    "price": 95
  },
  {
    "item_id": "SG003",
    "name": "Mystique",
    "description": "Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.",
    "quantity_in_stock": 3,
    "price": 70
  },
  {
    "item_id": "SG004",
    "name": "Sport",
    "description": "Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.",
    "quantity_in_stock": 11,
    "price": 110
  },
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 10,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]

Transactions Table
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-04T14:11:03.974192"
  }
]
"""

# As you can see above, the schemas of each table are as follows:
# 
# <div style="border:1px solid #BFDBFE; border-left:6px solid #3B82F6; background:#EFF6FF; border-radius:6px; padding:16px; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif; line-height:1.6; color:#1E3A8A;">
# 
#   <h4 style="margin-top:0; color:#1E40AF;">Inventory Table (<code>inventory_tbl</code>)</h4>
#   <ul>
#     <li><strong>item_id</strong> (string): Unique product identifier (e.g., SG001).</li>
#     <li><strong>name</strong> (string): Style of sunglasses (e.g., Aviator, Round).</li>
#     <li><strong>description</strong> (string): Text description of the product.</li>
#     <li><strong>quantity_in_stock</strong> (int): Current stock available.</li>
#     <li><strong>price</strong> (float): Price in USD.</li>
#   </ul>
#   <h4 style="margin-top:1em; color:#1E40AF;">Transactions Table (<code>transactions_tbl</code>)</h4>
#   <ul>
#     <li><strong>transaction_id</strong> (string): Unique identifier (e.g., TXN001).</li>
#     <li><strong>customer_name</strong> (string): Name of the customer, or <code>OPENING_BALANCE</code> for initial entry.</li>
#     <li><strong>transaction_summary</strong> (string): Short description of the transaction.</li>
#     <li><strong>transaction_amount</strong> (float): Amount of money for this transaction.</li>
#     <li><strong>balance_after_transaction</strong> (float): Running balance after applying the transaction.</li>
#     <li><strong>timestamp</strong> (string): ISO-8601 formatted date/time of the transaction.</li>
#   </ul>
# </div>
# 

# ## Planning with Code Execution

# ### 2.1. The plan
# 
# Once the schema is clear, you’ll build the **prompt** that instructs the model to *plan by writing code* and then execute that code. As Andrew emphasized, the code is the plan: the model explains each step in comments, then carries it out. Your prompt below also makes the model self-decide whether the request is read-only or a state change, and it enforces safe execution (no I/O, no network, TinyDB Query only, consistent mutations).
# 

# In[4]:


PROMPT = """You are a senior data assistant. PLAN BY WRITING PYTHON CODE USING TINYDB.

Database Schema & Samples (read-only):
{schema_block}

Execution Environment (already imported/provided):
- Variables: db, inventory_tbl, transactions_tbl  # TinyDB Table objects
- Helpers: get_current_balance(tbl) -> float, next_transaction_id(tbl, prefix="TXN") -> str
- Natural language: user_request: str  # the original user message

PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request (shape/keywords, price ranges "under/over/between", stock mentions,
  quantities, buy/return intent). Do NOT hard-code values.
- Build TinyDB queries dynamically with Query(). If a constraint isn't in user_request, don't apply it.
- Be conservative: if intent is ambiguous, do read-only (DRY RUN).

TRANSACTION POLICY (hard):
- Do NOT create aggregated multi-item transactions.
- If the request contains multiple items, create a separate transaction row PER ITEM.
- For each item:
  - compute its own line total (unit_price * qty),
  - insert ONE transaction with that amount,
  - update balance sequentially (balance += line_total),
  - update the item’s stock.
- If any requested item lacks sufficient stock, do NOT mutate anything; reply with STATUS="insufficient_stock".

HUMAN RESPONSE REQUIREMENT (hard):
- You MUST set a variable named `answer_text` (type str) with a short, customer-friendly sentence (1–2 lines).
- This sentence is the only user-facing message. No dataframes/JSON, no boilerplate disclaimers.
- If nothing matches, politely say so and offer a nearby alternative (closest style/price) or a next step.

ACTION POLICY:
- If the request clearly asks to change state (buy/purchase/return/restock/adjust):
    ACTION="mutate"; SHOULD_MUTATE=True; perform the change and write a matching transaction row.
  Otherwise:
    ACTION="read"; SHOULD_MUTATE=False; simulate and explain briefly as a dry run (in logs only).

FAILURE & EDGE-CASE HANDLING (must implement):
- Do not capture outer variables in Query.test. Pass them as explicit args.
- Always set a short `answer_text`. Also set a string `STATUS` to one of:
  "success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent".
- no_match: No items satisfy the filters → suggest the closest in style/price, or invite a different range.
- insufficient_stock: Item found but stock < requested qty → state available qty and offer the max you can fulfill.
- invalid_request: Unable to parse essential info (e.g., quantity for a purchase/return) → ask for the missing piece succinctly.
- unsupported_intent: The action is outside the store’s capabilities → provide the nearest supported alternative.
- In all cases, keep the tone helpful and concise (1–2 sentences). Put technical details (e.g., ACTION/DRY RUN) only in stdout logs.

OUTPUT CONTRACT:
- Return ONLY executable Python between these tags (no extra text):
  <execute_python>
  # your python
  </execute_python>

CODE CHECKLIST (follow in code):
1) Parse intent & constraints from user_request (regex ok).
2) Build TinyDB condition incrementally; query inventory_tbl.
3) If mutate: validate stock, update inventory, insert a transaction (new id, amount, balance, timestamp).
4) ALWAYS set:
   - `answer_text` (human sentence, required),
   - `STATUS` (see list above).
   Also print a brief log to stdout, e.g., "LOG: ACTION=read DRY_RUN=True STATUS=no_match".
5) Optional: set `answer_rows` or `answer_json` if useful, but `answer_text` is mandatory.

TONE EXAMPLES (for `answer_text`):
- success: "Yes, we have our Classic sunglasses, a round frame, for $60."
- no_match: "We don’t have round frames under $100 in stock right now, but our Moon round frame is available at $120."
- insufficient_stock: "We only have 1 pair of Classic left; I can reserve that for you."
- invalid_request: "I can help with that—how many pairs would you like to purchase?"
- unsupported_intent: "We can’t refurbish frames, but I can suggest similar new models."

Constraints:
- Use TinyDB Query for filtering. Standard library imports only if needed.
- Keep code clear and commented with numbered steps.

User request:
{question}
"""


# ### 2.2 From Prompt to Code (Planning in Code)
# 
# Let’s generate code that **is the plan**.
# 
# Instead of asking the model to output a plan in JSON and running it step-by-step with many tiny tools, let’s have it **write Python that encodes the whole plan** (e.g., “filter this, then compute that, then update this row”). The function `generate_llm_code`:
# 
# 1. **Builds a live schema** from `inventory_tbl` and `transactions_tbl` so the model sees real fields, types, and examples.
# 2. **Formats the prompt** with that schema plus the user’s question.
# 3. **Calls the model** to produce a **plan-with-code** response — typically an `<execute_python>...</execute_python>` block whose body contains the step-by-step logic.
# 4. **Returns the full response** (including the plan and the code).  
#    *We don’t execute anything in this step.*
# 
# Why this pattern? Let’s leverage Python/TinyDB as a rich toolbox the model already “knows,” so it can compose multi-step solutions directly in code instead of relying on a growing set of bespoke tools. We’ll extract and run the code in a later step.

# In[5]:


# ---------- 1) Code generation ----------
def generate_llm_code(
    prompt: str,
    *,
    inventory_tbl,
    transactions_tbl,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> str:
    """
    Ask the LLM to produce a plan-with-code response.
    Returns the FULL assistant content (including surrounding text and tags).
    The actual code extraction happens later in execute_generated_code.
    """
    schema_block = inv_utils.build_schema_block(inventory_tbl, transactions_tbl)
    prompt = PROMPT.format(schema_block=schema_block, question=prompt)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You write safe, well-commented TinyDB code to handle data questions and updates."
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    
    return content  


# ### 2.3 Try a Sample Prompt (Planning-in-Code)
# 
# We’ll use the same prompt Andrew used in the lecture:
# 
# > **Prompt:** “Do you have any round sunglasses in stock that are under $100?”
# 
# Before generating any code, let’s manually inspect the TinyDB tables to see if there are truly *round* frames (word-only match) and what their prices look like. Run the next cell to preview the inventory and highlight items that match the word-only “round” filter.

# In[6]:


Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)

# Search the inventory table for documents where either the description OR the name
# contains the word "round" (case-insensitive). The check is done inline:
# - (v or "") ensures we handle None by converting it to an empty string
# - .lower() normalizes case
# - " round " enforces a crude word boundary (won't match "wraparound")
round_sunglasses = inventory_tbl.search(
    (Item.description.test(lambda v: " round " in ((v or "").lower()))) |
    (Item.name.test(        lambda v: " round " in ((v or "").lower())))
)

# Render the results as formatted JSON in the notebook UI
utils.print_html(json.dumps(round_sunglasses, indent=2), title="Inventory Status: Round Sunglasses")
"""
Inventory Status: Round Sunglasses
[
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 10,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]
"""

# Great — we do have round frames available. From our manual inspection, there are two round styles in stock, but only **one** is **under \$100**. Therefore, the item that satisfies the requirement is:
# 
# ````python
# {
#   "item_id": "SG005",
#   "name": "Classic",
#   "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
#   "quantity_in_stock": 10,
#   "price": 60
# }
# ````
# 
# Now let’s ask the model to **generate a plan in code** that answers Andrew’s prompt (no execution yet).

# In[7]:


# Andrew's prompt from the lecture
prompt_round = "Do you have any round sunglasses in stock that are under $100?"

# Generate the plan-as-code (FULL content; may include <execute_python> tags)
full_content_round = generate_llm_code(
    prompt_round,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1.0,
)

# Inspect the LLM’s plan + code (no execution here)
utils.print_html(full_content_round, title="Plan with Code (Full Response)")
"""
Plan with Code (Full Response)
<execute_python>
# 1) Parse user_request for filters
import re
from tinydb import Query

# Assuming user_request is defined
request = user_request.lower()

# Determine if the user is asking about "round" sunglasses
shape_keyword = None
if "round" in request:
    shape_keyword = "round"

# Determine price constraints ("under $100")
price_upper = None
match = re.search(r'under\s*\$(\d+)', request)
if match:
    price_upper = float(match.group(1))

# 2) Build TinyDB query for read-only search
Item = Query()
conditions = []
# stock > 0
conditions.append(Item.quantity_in_stock.test(lambda x: x > 0))
# shape filter on description or name
if shape_keyword:
    kw = shape_keyword
    conditions.append(
        (Item.description.test(lambda desc, kw=kw: kw in desc.lower())) |
        (Item.name.test(lambda name, kw=kw: kw in name.lower()))
    )
# price upper bound
if price_upper is not None:
    pu = price_upper
    conditions.append(Item.price.test(lambda p, pu=pu: p < pu))

# Combine all conditions
from functools import reduce
from operator import and_
if conditions:
    query_expr = reduce(and_, conditions)
    results = inventory_tbl.search(query_expr)
else:
    results = inventory_tbl.all()

# 3) Prepare response
if results:
    # List matching items with name and price
    items_str = []
    for item in results:
        items_str.append(f"{item['name']} at ${item['price']}")
    # Join with commas and 'and' for readability
    if len(items_str) == 1:
        items_list = items_str[0]
    else:
        items_list = ", ".join(items_str[:-1]) + " and " + items_str[-1]
    answer_text = f"Yes, we have {items_list} in stock."
    STATUS = "success"
else:
    # No matches: suggest nearest alternative by price ascending
    # Find any round items in stock regardless of price
    alt_conditions = [
        Item.quantity_in_stock.test(lambda x: x > 0),
        (Item.description.test(lambda d, kw="round": kw in d.lower())) |
        (Item.name.test(lambda n, kw="round": kw in n.lower()))
    ]
    alt_query = reduce(and_, alt_conditions)
    alternatives = sorted(inventory_tbl.search(alt_query), key=lambda x: x['price'])
    if alternatives:
        alt = alternatives[0]
        answer_text = (f"We don’t have round frames under ${int(price_upper)} in stock right now, "
                       f"but our {alt['name']} round frame is available at ${alt['price']}.")
    else:
        answer_text = "We don’t have any round sunglasses in stock at the moment; can I help you find something else?"
    STATUS = "no_match"

# 4) Log action
print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
</execute_python>
"""

# ### 2.4. Define the executor function (run a given plan)
# 
# Now we’ll define the function that **takes a plan produced by the model and runs it** safely:
# 
# - It **accepts either** the full LLM response (with `<execute_python>…</execute_python>`) **or** raw Python code.
# - It **extracts** the executable block when needed.
# - It runs the code in a **controlled namespace** (TinyDB tables + safe helpers only).
# - It captures **stdout**, **errors**, and the model-set answer variables (`answer_text`, `answer_rows`, `answer_json`).
# - It renders **before/after** table snapshots to make side effects explicit.
# 
# This is the “executor” that turns a **plan-as-code** into actions and a concise user-facing answer.
# 

# In[8]:


# --- Helper: extract code between <execute_python>...</execute_python> ---
def _extract_execute_block(text: str) -> str:
    """
    Returns the Python code inside <execute_python>...</execute_python>.
    If no tags are found, assumes 'text' is already raw Python code.
    """
    if not text:
        raise RuntimeError("Empty content passed to code executor.")
    m = re.search(r"<execute_python>(.*?)</execute_python>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ---------- 2) Code execution ----------
def execute_generated_code(
    code_or_content: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute code in a controlled namespace.
    Accepts either raw Python code OR full content with <execute_python> tags.
    Returns minimal artifacts: stdout, error, and extracted answer.
    """
    # Extract code here (now centralized)
    code = _extract_execute_block(code_or_content)

    SAFE_GLOBALS = {
        "Query": Query,
        "get_current_balance": inv_utils.get_current_balance,
        "next_transaction_id": inv_utils.next_transaction_id,
        "user_request": user_request or "",
    }
    SAFE_LOCALS = {
        "db": db,
        "inventory_tbl": inventory_tbl,
        "transactions_tbl": transactions_tbl,
    }

    # Capture stdout from the executed code
    _stdout_buf, _old_stdout = io.StringIO(), sys.stdout
    sys.stdout = _stdout_buf
    err_text = None
    try:
        exec(code, SAFE_GLOBALS, SAFE_LOCALS)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = _old_stdout
    printed = _stdout_buf.getvalue().strip()

    # Extract possible answers set by the generated code
    answer = (
        SAFE_LOCALS.get("answer_text")
        or SAFE_LOCALS.get("answer_rows")
        or SAFE_LOCALS.get("answer_json")
    )


    return {
        "code": code,            # <- ya sin etiquetas
        "stdout": printed,
        "error": err_text,
        "answer": answer,
        "transactions_tbl": transactions_tbl.all(),  # For inspection
        "inventory_tbl": inventory_tbl.all(),  # For inspection
    }


# You’ve checked the shelves and confirmed there’s exactly one round style under $100. Now the fun part: let’s hand the model’s plan-as-code to our executor and watch it do the work. The executor will peel out the <code><execute_python>...</execute_python></code> block, run it in a locked-down sandbox, and then show you everything that matters—what changed in the tables (before/after), any logs the plan printed, and the final, customer-friendly answer_text.

# In[9]:


# Execute the generated plan for the round-sunglasses question
result = execute_generated_code(
    full_content_round,          # the full LLM response you generated earlier
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    user_request=prompt_round, # e.g., "Do you have any round sunglasses in stock that are under $100?"
)

# Peek at exactly what Python the plan executed
utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
"""
Plan Execution · Extracted Answer
Yes, we have Classic at $60 in stock.
"""

# As you can see, this is the expected result based on our previous manual analysis.

# ## 2.4 Return Two Aviator Sunglasses
# 
# In the previous step we only **queried** the data, so inventory and transactions were unchanged.  
# Now let’s handle a **return** scenario using the planning-in-code pattern:
# > **Request:** “Return 2 Aviator sunglasses I bought last week.”
# 
# Before generating the plan, let’s **inspect the current inventory** for the *Aviator* model.

# In[10]:


Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)

# Query: fetch all inventory rows whose 'name' is exactly "Aviator".
# Notes:
# - This is a case-sensitive equality check. "aviator" won't match.
# - If you need case-insensitive matching, consider a .test(...) or .matches(...) with re.I.
aviators = inventory_tbl.search(
    (Item.name == "Aviator")
)

# Display the matched documents in a readable JSON panel
utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses before return")
"""
Inventory status: Aviator sunglasses before return
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 23,
    "price": 80
  }
]
"""

# Inventory confirms one Aviator SKU in stock — **SG001 (Aviator)**: **23** units at **$80** each. Now let's generate a plan to answer the prompt:

# In[11]:


prompt_aviator = "Return 2 Aviator sunglasses I bought last week."

# Generate the plan-as-code (FULL content; may include <execute_python> tags)
full_content_aviator = generate_llm_code(
    prompt_aviator,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1,
)

# Inspect the LLM’s plan + code (no execution here)
utils.print_html(full_content_aviator, title="Plan with Code (Full Response)")
"""
Plan with Code (Full Response)
<execute_python>
import re
from datetime import datetime
from tinydb import Query

# Step 1: Parse quantity from user_request
qty_match = re.search(r'(\d+)', user_request)
if not qty_match:
    STATUS = "invalid_request"
    answer_text = "I can help with that—how many pairs would you like to return?"
    print(f"LOG: ACTION=unsupported_intent STATUS={STATUS}")
else:
    qty = int(qty_match.group(1))
    # Step 2: Identify the item name by matching inventory names in user_request
    all_items = inventory_tbl.all()
    found_names = [item['name'] for item in all_items 
                   if item['name'].lower() in user_request.lower()]
    if len(found_names) != 1:
        STATUS = "no_match"
        # Suggest the closest priced item if possible
        STATUS = "no_match"
        # Find any item as alternative (e.g., cheapest)
        alt_item = min(all_items, key=lambda x: x['price'])
        answer_text = (f"We don’t have that exact style to return, but our {alt_item['name']} is "
                       f"available for ${alt_item['price']}.")
        print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
    else:
        item_name = found_names[0]
        # Step 3: Query the inventory_tbl for the item
        query = Query()
        def match_name(val, nl): return val.lower() == nl
        name_lower = item_name.lower()
        results = inventory_tbl.search(query.name.test(match_name, name_lower))
        if not results:
            STATUS = "no_match"
            answer_text = f"We don’t carry {item_name} sunglasses."
            print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
        else:
            # We have exactly one matching item
            item = results[0]
            # Step 4: Process the return (mutation)
            STATUS = "success"
            ACTION = "mutate"
            SHOULD_MUTATE = True
            # Update stock: add returned qty
            new_stock = item['quantity_in_stock'] + qty
            inventory_tbl.update({'quantity_in_stock': new_stock}, query.item_id == item['item_id'])
            # Prepare transaction record
            price = item['price']
            refund_amount = - price * qty  # negative for refund
            current_balance = get_current_balance(transactions_tbl)
            new_balance = current_balance + refund_amount
            txn_id = next_transaction_id(transactions_tbl, prefix="TXN")
            txn = {
                'transaction_id': txn_id,
                'customer_name': 'RETURN',
                'transaction_summary': f"Return of {qty} {item_name} sunglasses",
                'transaction_amount': refund_amount,
                'balance_after_transaction': new_balance,
                'timestamp': datetime.now().isoformat()
            }
            transactions_tbl.insert(txn)
            answer_text = (f"Your return of {qty} {item_name} sunglasses has been processed; "
                           f"a refund of ${-refund_amount} has been issued.")
            print(f"LOG: ACTION={ACTION} STATUS={STATUS}")
</execute_python>
"""

# Before we execute the plan, let’s check the current status of the transactions.

# In[12]:


utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table Before Return")
"""
Transactions Table Before Return
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-04T14:11:03.974192"
  }
]
"""

# The transaction log currently shows a single entry — the opening balance (`TXN001`) for `$500.00` recorded at `2025-10-03T09:16:59.628898`. 
# 
# Ready to go—execute the plan by running the cell below.

# In[13]:


# Execute the generated plan for the round-sunglasses question
result = execute_generated_code(
    full_content_aviator,          # the full LLM response you generated earlier
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    user_request=prompt_aviator, # e.g., "Return 2 aviator sunglasses I bought last week."
)

# Peek at exactly what Python the plan executed
utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
"""
Plan Execution · Extracted Answer
Your return of 2 Aviator sunglasses has been processed; a refund of $160 has been issued.
"""

# You can see below that a new transaction has been inserted for the Aviator sunglasses return.

# In[14]:


utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table After Return")
"""
Transactions Table After Return
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-04T14:11:03.974192"
  },
  {
    "transaction_id": "TXN002",
    "customer_name": "RETURN",
    "transaction_summary": "Return of 2 Aviator sunglasses",
    "transaction_amount": -160,
    "balance_after_transaction": 340.0,
    "timestamp": "2026-03-04T14:11:33.301162"
  }
]
"""

# And by running the cell below, you’ll see the Aviator stock increase to 25 (`quantity_in_stock`).

# In[15]:


Item = Query()                  

aviators = inventory_tbl.search(
    (Item.name == "Aviator")
)

utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses after return")
"""
Inventory status: Aviator sunglasses after return
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 25,
    "price": 80
  }
]
"""

# ## 3. Putting It All Together: Customer Service Agent
# 
# You’ve built the pieces—schema, prompt, code generator, and executor. Now let’s wire them up into a single helper that takes a natural-language request, generates a plan-as-code, executes it safely, and shows the result (plus before/after tables).
# 
# **What this agent does**
# - Optionally reseeds the demo data for a clean run.
# - Generates the plan (Python inside `<execute_python>…</execute_python>`).
# - Executes the plan in a controlled namespace (TinyDB + helpers).
# - Surfaces a concise `answer_text` and renders before/after snapshots.

# In[16]:


def customer_service_agent(
    question: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    model: str = "o4-mini",
    temperature: float = 1.0,
    reseed: bool = False,
) -> dict:
    """
    End-to-end helper:
      1) (Optional) reseed inventory & transactions
      2) Generate plan-as-code from `question`
      3) Execute in a controlled namespace
      4) Render before/after snapshots and return artifacts

    Returns:
      {
        "full_content": <raw LLM response (may include <execute_python> tags)>,
        "exec": {
            "code": <extracted python>,
            "stdout": <plan logs>,
            "error": <traceback or None>,
            "answer": <answer_text/rows/json>,
            "inventory_after": [...],
            "transactions_after": [...]
        }
      }
    """
    # 0) Optional reseed
    if reseed:
        inv_utils.create_inventory()
        inv_utils.create_transactions()

    # 1) Show the question
    utils.print_html(question, title="User Question")

    # 2) Generate plan-as-code (FULL content)
    full_content = generate_llm_code(
        question,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
    )
    utils.print_html(full_content, title="Plan with Code (Full Response)")

    # 3) Before snapshots
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · Before")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · Before")

    # 4) Execute
    exec_res = execute_generated_code(
        full_content,
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        user_request=question,
    )

    # 5) After snapshots + final answer
    utils.print_html(exec_res["answer"], title="Plan Execution · Extracted Answer")
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · After")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · After")

    # 6) Return artifacts
    return {
        "full_content": full_content,
        "exec": {
            "code": exec_res["code"],
            "stdout": exec_res["stdout"],
            "error": exec_res["error"],
            "answer": exec_res["answer"],
            "inventory_after": inventory_tbl.all(),
            "transactions_after": transactions_tbl.all(),
        },
    }


# ## 4. Try It Out (with the Customer Service Agent)
# 
# Use the `customer_service_agent(...)` helper to go from a natural-language request → plan-as-code → safe execution → before/after snapshots.
# 
# **Try these prompts:**
# 1) **Read-only (Andrew’s example):**  
#    “Do you have any round sunglasses in stock that are under $100?”
# 2) **Mutation — return:**  
#    “Return 2 Aviator sunglasses.”
# 3) **Mutation — purchase:**  
#    “Purchase 3 Wayfarer sunglasses for customer Alice.”
# 4) **Mutation - purchase multiple items:**
#    "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator."
# 
# 
# <div style="border:1px solid #93c5fd; border-left:6px solid #3b82f6; background:#eff6ff; border-radius:8px; padding:14px 16px; color:#1e3a8a; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
#   🔎 <strong>What does <code>reseed=True</code> do?</strong><br><br>
#   When you call <code>customer_service_agent(..., reseed=True)</code>, the agent <em>re-initializes</em> the demo data before running your prompt:
#   <ul style="margin:8px 0 0 18px;">
#     <li><strong>Resets</strong> the <code>inventory_tbl</code> to the default product set.</li>
#     <li><strong>Resets</strong> the <code>transactions_tbl</code> to a single opening-balance entry.</li>
#     <li>Ensures a <strong>clean, reproducible</strong> run so results aren’t affected by previous tests.</li>
#   </ul>
#   Set <code>reseed=False</code> if you want to <strong>preserve</strong> the current state and continue from prior operations.
# </div>
# 
# 

# In[17]:


prompt = "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator sunglasses."

out = customer_service_agent(
    prompt,
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1.0,
    reseed=True,   # set False to keep current state of the inventory and the transactions
)
"""
User Question
I want to buy 3 pairs of classic sunglasses and 1 pair of aviator sunglasses.

Plan with Code (Full Response)
<execute_python>
# 1) imports
from tinydb import Query
import re
from datetime import datetime

# 2) initialize
text = user_request

# 3) detect purchase intent
if not re.search(r"\b(buy|purchase)\b", text, re.IGNORECASE):
    ACTION="read"; SHOULD_MUTATE=False; STATUS="unsupported_intent"
    answer_text="I can help with browsing our inventory; what are you interested in?"
    print(f"LOG: ACTION={ACTION} DRY_RUN={not SHOULD_MUTATE} STATUS={STATUS}")
else:
    # 4) parse items and quantities
    pattern = re.compile(r"(\d+)\s+pairs?\s+of\s+([A-Za-z]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        ACTION="read"; SHOULD_MUTATE=False; STATUS="invalid_request"
        answer_text="I can help with that—how many pairs would you like to purchase?"
        print(f"LOG: ACTION={ACTION} DRY_RUN={not SHOULD_MUTATE} STATUS={STATUS}")
    else:
        # Normalize item names and quantities
        items = [(m[1].capitalize(), int(m[0])) for m in matches]
        # 5) check stock availability
        Inventory = Query()
        insufficient = None
        for name, qty in items:
            rows = inventory_tbl.search(Inventory.name.test(lambda v, n=name: v.lower()==n.lower()))
            if not rows:
                insufficient = (name, 0)
                break
            stock = rows[0]['quantity_in_stock']
            if stock < qty:
                insufficient = (name, stock)
                break
        if insufficient:
            ACTION="read"; SHOULD_MUTATE=False; STATUS="insufficient_stock"
            name, stock = insufficient
            if stock > 0:
                answer_text = f"We only have {stock} pair{'s' if stock!=1 else ''} of {name}; I can reserve that for you."
            else:
                answer_text = f"Sorry, we don't have any {name} in stock right now."
            print(f"LOG: ACTION={ACTION} DRY_RUN={not SHOULD_MUTATE} STATUS={STATUS}")
        else:
            # 6) process each purchase
            ACTION="mutate"; SHOULD_MUTATE=True
            current_balance = get_current_balance(transactions_tbl)
            for name, qty in items:
                row = inventory_tbl.search(Inventory.name.test(lambda v, n=name: v.lower()==n.lower()))[0]
                unit_price = row['price']
                line_total = unit_price * qty
                # update stock
                inventory_tbl.update(
                    {'quantity_in_stock': row['quantity_in_stock'] - qty},
                    Inventory.item_id == row['item_id']
                )
                # insert transaction
                txn_id = next_transaction_id(transactions_tbl, prefix="TXN")
                current_balance += line_total
                transactions_tbl.insert({
                    'transaction_id': txn_id,
                    'customer_name': 'CUSTOMER',
                    'transaction_summary': f'Purchase of {qty} {name} sunglasses',
                    'transaction_amount': line_total,
                    'balance_after_transaction': current_balance,
                    'timestamp': datetime.now().isoformat()
                })
            STATUS="success"
            # Build friendly confirmation
            items_desc = " and ".join(f"{qty} {name}" for name, qty in items)
            answer_text = f"Your order for {items_desc} sunglasses is confirmed!"
            print(f"LOG: ACTION={ACTION} DRY_RUN={not SHOULD_MUTATE} STATUS={STATUS}")
</execute_python>

Inventory Table · Before
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 23,
    "price": 80
  },
  {
    "item_id": "SG002",
    "name": "Wayfarer",
    "description": "Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.",
    "quantity_in_stock": 6,
    "price": 95
  },
  {
    "item_id": "SG003",
    "name": "Mystique",
    "description": "Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.",
    "quantity_in_stock": 3,
    "price": 70
  },
  {
    "item_id": "SG004",
    "name": "Sport",
    "description": "Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.",
    "quantity_in_stock": 11,
    "price": 110
  },
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 10,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]

Transactions Table · Before
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-04T14:11:33.452330"
  }
]

Plan Execution · Extracted Answer
Your order for 3 Classic and 1 Aviator sunglasses is confirmed!


Inventory Table · After
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 22,
    "price": 80
  },
  {
    "item_id": "SG002",
    "name": "Wayfarer",
    "description": "Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.",
    "quantity_in_stock": 6,
    "price": 95
  },
  {
    "item_id": "SG003",
    "name": "Mystique",
    "description": "Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.",
    "quantity_in_stock": 3,
    "price": 70
  },
  {
    "item_id": "SG004",
    "name": "Sport",
    "description": "Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.",
    "quantity_in_stock": 11,
    "price": 110
  },
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 7,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]


Transactions Table · After
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-04T14:11:33.452330"
  },
  {
    "transaction_id": "TXN002",
    "customer_name": "CUSTOMER",
    "transaction_summary": "Purchase of 3 Classic sunglasses",
    "transaction_amount": 180,
    "balance_after_transaction": 680.0,
    "timestamp": "2026-03-04T14:11:55.497551"
  },
  {
    "transaction_id": "TXN003",
    "customer_name": "CUSTOMER",
    "transaction_summary": "Purchase of 1 Aviator sunglasses",
    "transaction_amount": 80,
    "balance_after_transaction": 760.0,
    "timestamp": "2026-03-04T14:11:55.836929"
  }
]
"""

# ## 5. Takeaways
# 
# - **You let code be the plan.** Following Andrew’s “code-as-action” idea, you had the model write Python that chains the steps (filter → compute → update) and then you just ran it.
# 
# - **You skipped the brittle tool soup.** Instead of piling on tiny tools or JSON plans, you used Python/TinyDB—giving the model a big, familiar toolbox that handles many query shapes with one prompt.
# 
# - **You kept runs safe and visible.** You executed in a controlled namespace, captured logs/errors, and reviewed before/after tables—so you always know what changed and why.

# <div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
# 
# 🎉 <strong>Congratulations!</strong>
# 
# You just finished the lab and built an <em>agentic</em> customer service workflow. You let the model write code as the plan, ran it safely, and used simple validations to keep updates reliable. When things failed, you surfaced clear, human-readable reasons; when things worked, you saw exactly what changed via before/after snapshots.
# 
# With this pattern—planning <em>in</em> code, plus transparent execution—you’re ready to design your own workflows that feel automatic, safe, and easy to extend. 🚀
# 
# </div>
# 

# My own example:
prompt = "I want to exchange the Wayfarer sunglasses I bought last week to Mystique sunglasses."

out = customer_service_agent(
    prompt,
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1.0,
    reseed=True,   # set False to keep current state of the inventory and the transactions
)
"""
User Question
I want to exchange the Wayfarer sunglasses I bought last week to Mystique sunglasses.

Plan with Code (Full Response)
<execute_python>
from tinydb import Query
from datetime import datetime

# Step 1: Initialize and parse intent (exchange), extract source and target item names
request = user_request.lower()
import re

# Default STATUS
STATUS = "invalid_request"

# Try to extract "exchange X to Y"
match = re.search(r"exchange\s+the\s+(.+?)\s+sunglasses.*to\s+(.+?)\s+sunglasses", request)
if not match:
    # Ask for clarity if parsing fails
    answer_text = "I can help with that—please specify which sunglasses you'd like to exchange from and to."
    STATUS = "invalid_request"
    print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
else:
    # Extract item names
    source_name = match.group(1).strip()
    target_name = match.group(2).strip()
    qty = 1  # default for exchange is one pair
    
    # Step 2: Lookup source and target in inventory
    Item = Query()
    source_rows = inventory_tbl.search(Item.name.test(lambda v, s=source_name: v.lower() == s.lower()))
    target_rows = inventory_tbl.search(Item.name.test(lambda v, t=target_name: v.lower() == t.lower()))
    
    # If either not found
    if not source_rows or not target_rows:
        # No match for one of the styles
        STATUS = "no_match"
        # Suggest alternative if source missing or target missing
        missing = source_name if not source_rows else target_name
        answer_text = f"Sorry, we don’t have {missing} sunglasses in stock right now."
        print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
    else:
        source = source_rows[0]
        target = target_rows[0]
        # Check target stock availability
        if target['quantity_in_stock'] < qty:
            STATUS = "insufficient_stock"
            answer_text = f"We only have {target['quantity_in_stock']} of {target_name} left; let me know if you'd like to reserve them."
            print(f"LOG: ACTION=read DRY_RUN=True STATUS={STATUS}")
        else:
            # Step 3: Perform mutation: return source then purchase target
            # Get current balance
            balance = get_current_balance(transactions_tbl)
            
            # Return transaction
            return_amount = source['price'] * qty
            return_txn_id = next_transaction_id(transactions_tbl)
            balance_after_return = balance + return_amount
            transactions_tbl.insert({
                'transaction_id': return_txn_id,
                'customer_name': 'EXCHANGE',
                'transaction_summary': f"Return {source['name']} sunglasses",
                'transaction_amount': return_amount,
                'balance_after_transaction': balance_after_return,
                'timestamp': datetime.now().isoformat()
            })
            # Update source stock
            inventory_tbl.update({'quantity_in_stock': source['quantity_in_stock'] + qty},
                                 Item.item_id == source['item_id'])
            
            # Purchase transaction
            purchase_amount = - target['price'] * qty
            purchase_txn_id = next_transaction_id(transactions_tbl)
            balance_after_purchase = balance_after_return + purchase_amount
            transactions_tbl.insert({
                'transaction_id': purchase_txn_id,
                'customer_name': 'EXCHANGE',
                'transaction_summary': f"Purchase {target['name']} sunglasses",
                'transaction_amount': purchase_amount,
                'balance_after_transaction': balance_after_purchase,
                'timestamp': datetime.now().isoformat()
            })
            # Update target stock
            inventory_tbl.update({'quantity_in_stock': target['quantity_in_stock'] - qty},
                                 Item.item_id == target['item_id'])
            
            STATUS = "success"
            answer_text = (f"Your exchange of {source['name']} to {target['name']} is complete; "
                           f"your updated balance is ${balance_after_purchase:.2f}.")
            print(f"LOG: ACTION=mutate STATUS={STATUS}")
# answer_text and STATUS are set for user response
</execute_python>

Inventory Table · Before
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 23,
    "price": 80
  },
  {
    "item_id": "SG002",
    "name": "Wayfarer",
    "description": "Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.",
    "quantity_in_stock": 6,
    "price": 95
  },
  {
    "item_id": "SG003",
    "name": "Mystique",
    "description": "Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.",
    "quantity_in_stock": 3,
    "price": 70
  },
  {
    "item_id": "SG004",
    "name": "Sport",
    "description": "Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.",
    "quantity_in_stock": 11,
    "price": 110
  },
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 10,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]

Transactions Table · Before
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-05T13:36:02.064740"
  }
]

Plan Execution · Extracted Answer
Your exchange of Wayfarer to Mystique is complete; your updated balance is $525.00.

Inventory Table · After
[
  {
    "item_id": "SG001",
    "name": "Aviator",
    "description": "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames offer timeless appeal. The large lenses provide excellent coverage while the lightweight construction ensures comfort during long wear.",
    "quantity_in_stock": 23,
    "price": 80
  },
  {
    "item_id": "SG002",
    "name": "Wayfarer",
    "description": "Featuring thick, angular frames that make a statement, these sunglasses combine retro charm with modern edge. The rectangular lenses and sturdy acetate construction create a confident look.",
    "quantity_in_stock": 7,
    "price": 95
  },
  {
    "item_id": "SG003",
    "name": "Mystique",
    "description": "Inspired by 1950s glamour, these frames sweep upward at the outer corners to create an elegant, feminine silhouette. The subtle curves and often embellished temples add sophistication to any outfit.",
    "quantity_in_stock": 2,
    "price": 70
  },
  {
    "item_id": "SG004",
    "name": "Sport",
    "description": "Designed for active lifestyles, these wraparound sunglasses feature a single curved lens that provides maximum coverage and wind protection. The lightweight, flexible frames include rubber grips.",
    "quantity_in_stock": 11,
    "price": 110
  },
  {
    "item_id": "SG005",
    "name": "Classic",
    "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
    "quantity_in_stock": 10,
    "price": 60
  },
  {
    "item_id": "SG006",
    "name": "Moon",
    "description": "Oversized round style with bold plastic frames, evoking retro aesthetics with a modern twist.",
    "quantity_in_stock": 10,
    "price": 120
  }
]

Transactions Table · After
[
  {
    "transaction_id": "TXN001",
    "customer_name": "OPENING_BALANCE",
    "transaction_summary": "Daily opening register balance",
    "transaction_amount": 500.0,
    "balance_after_transaction": 500.0,
    "timestamp": "2026-03-05T13:36:02.064740"
  },
  {
    "transaction_id": "TXN002",
    "customer_name": "EXCHANGE",
    "transaction_summary": "Return Wayfarer sunglasses",
    "transaction_amount": 95,
    "balance_after_transaction": 595.0,
    "timestamp": "2026-03-05T13:36:19.768348"
  },
  {
    "transaction_id": "TXN003",
    "customer_name": "EXCHANGE",
    "transaction_summary": "Purchase Mystique sunglasses",
    "transaction_amount": -70,
    "balance_after_transaction": 525.0,
    "timestamp": "2026-03-05T13:36:19.770320"
  }
]
"""
# My comments:
# 1. This is a second attempt to run the same prompt. The first attempt ended with "We don't handle explicit exchanges. Please return and then purchase."
# 2. Inventory table correctly recorded level movements: Wayfarer +1 to 7, Mystique -1 to 2.
# 3. Transaction table recorded two new transactions: Return Wayfarer and Purchase Mystique.
# 4. Transaction table recorded balance movements, which don't seem to be correct.
# Compare that to "Return of 2 Aviator sunglasses" example above.
# It correctly records inventory level movement +2 to 25 as well as the cash movement -160 to 340.
# So it associates "return" with "cash out".
# Compare that to "buy 3 pairs of classic sunglasses and 1 pair of aviator" example above.
# It correctly records both movements: inventory and cash.
# It seems that the word "exchange" somehow trips the agent off.



# In[18]:


import os
from IPython.display import FileLink
os.listdir('.') # list current directory


# In[19]:


FileLink('inv_utils.py')

