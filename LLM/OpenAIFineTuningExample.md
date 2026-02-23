```py
"""
Complete OpenAI Fine-Tuning Demo
=================================
Demonstrates the full SFT pipeline using OpenAI's Fine-Tuning API:

  Step 1  – Prepare and write training data as JSONL
  Step 2  – Upload the JSONL file to OpenAI
  Step 3  – Launch a fine-tuning job
  Step 4  – Poll until the job completes
  Step 5  – Run inference with the fine-tuned model
  Step 6  – Compare base vs fine-tuned model responses
  Step 7  – (Optional) Clean up — delete the file and model

Use case: teach GPT-4o-mini to respond as a formal banking assistant
that always structures answers with "Assessment:", "Recommendation:",
and "Risk Level:" — a style the base model doesn't use by default.

Install:
    pip install openai

Run:
    export OPENAI_API_KEY="sk-..."
    python openai_finetune_demo.py
"""

import os
import json
import time
import pathlib
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_MODEL     = "gpt-4o-mini-2024-07-18"   # cheapest model that supports SFT
TRAINING_FILE  = "banking_training.jsonl"
N_EPOCHS       = 3                           # how many passes over training data
POLL_INTERVAL  = 30                          # seconds between status checks

# ---------------------------------------------------------------------------
# STEP 1 — PREPARE TRAINING DATA
#
# Format: each line is a complete conversation in ChatML format.
# The "assistant" turn is what the model learns to produce.
#
# Rules:
#   - Minimum ~10 examples (OpenAI recommends 50-100+ for real tasks)
#   - Each example is one JSON object on a single line (JSONL)
#   - Must have "messages" key with role/content pairs
#   - system prompt is optional but recommended for consistency
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a formal banking assistant. Always structure your response with "
    "three labelled sections:\n"
    "Assessment: <your analysis>\n"
    "Recommendation: <your advice>\n"
    "Risk Level: <Low | Medium | High>"
)

TRAINING_EXAMPLES = [
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Should I invest my savings in stocks right now?"},
            {"role": "assistant", "content": (
                "Assessment: Current market volatility is elevated due to macroeconomic "
                "uncertainty. Equities have historically outperformed inflation over 10+ year "
                "horizons but carry short-term downside risk.\n"
                "Recommendation: Maintain a diversified portfolio aligned with your time horizon. "
                "Consider dollar-cost averaging rather than a lump-sum entry.\n"
                "Risk Level: Medium"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "I want to take out a personal loan to pay off my credit card debt."},
            {"role": "assistant", "content": (
                "Assessment: Consolidating high-interest credit card debt into a lower-rate "
                "personal loan is a recognised debt-management strategy. The benefit depends "
                "on the interest rate differential and your ability to avoid re-accumulating card debt.\n"
                "Recommendation: Obtain loan quotes from at least three lenders. Confirm the "
                "personal loan APR is materially lower than your current card rate before proceeding. "
                "Cancel or reduce credit limits post-consolidation to limit re-exposure.\n"
                "Risk Level: Low"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "My mortgage is up for renewal. Should I choose a fixed or variable rate?"},
            {"role": "assistant", "content": (
                "Assessment: Fixed rates provide payment certainty; variable rates may offer "
                "short-term savings but expose you to rate movements. Current central bank "
                "policy trajectory is a key factor in this decision.\n"
                "Recommendation: If your budget cannot absorb a 2-3% rate increase, select a "
                "fixed term. If you have financial flexibility and rates are expected to fall, "
                "a variable rate with a conversion option may be advantageous.\n"
                "Risk Level: Medium"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "How much should I keep in an emergency fund?"},
            {"role": "assistant", "content": (
                "Assessment: An emergency fund provides liquidity to cover unexpected expenses "
                "without incurring debt. The appropriate size depends on income stability, "
                "monthly obligations, and dependants.\n"
                "Recommendation: Maintain 3 to 6 months of essential living expenses in a "
                "high-yield savings account or money market fund. Households with variable "
                "income or single earners should target the higher end of this range.\n"
                "Risk Level: Low"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "I received an email asking for my banking details. What should I do?"},
            {"role": "assistant", "content": (
                "Assessment: Unsolicited requests for banking credentials are a primary vector "
                "for phishing and social engineering fraud. Legitimate financial institutions "
                "do not request passwords or full account numbers via email.\n"
                "Recommendation: Do not click any links or reply with personal information. "
                "Report the email to your bank's fraud team and delete it immediately. "
                "If you already disclosed credentials, contact your bank to freeze the account.\n"
                "Risk Level: High"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Is it a good idea to withdraw my pension early?"},
            {"role": "assistant", "content": (
                "Assessment: Early pension withdrawal typically triggers tax penalties "
                "(commonly 10% in the US for pre-59½ withdrawals) and permanently reduces "
                "compounding growth. It should be considered only in genuine financial hardship.\n"
                "Recommendation: Exhaust all alternatives — emergency fund, personal loan, "
                "hardship assistance programmes — before withdrawing pension funds. Consult "
                "a qualified financial adviser to model the long-term impact.\n"
                "Risk Level: High"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "What is the safest way to send a large international payment?"},
            {"role": "assistant", "content": (
                "Assessment: International wire transfers via SWIFT through your bank are "
                "the most regulated and traceable method. Third-party transfer services may "
                "offer lower fees but vary in regulatory oversight.\n"
                "Recommendation: Use your bank's official international wire service for "
                "amounts above $10,000. Verify recipient details through a secondary channel "
                "before initiating. Retain all transaction references for audit purposes.\n"
                "Risk Level: Medium"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Should I put extra money into my mortgage or invest it?"},
            {"role": "assistant", "content": (
                "Assessment: The optimal choice depends on your mortgage interest rate versus "
                "expected investment returns. Paying down mortgage debt offers a guaranteed "
                "return equal to the interest rate; investing carries market risk but "
                "historically yields higher long-term returns.\n"
                "Recommendation: If your mortgage rate exceeds 6%, prioritise additional "
                "repayments. Below that threshold, maximise tax-advantaged investment accounts "
                "first (401k, ISA), then split surplus between both goals.\n"
                "Risk Level: Low"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "My business is struggling with cash flow. What are my options?"},
            {"role": "assistant", "content": (
                "Assessment: Cash flow constraints in a business can stem from slow receivables, "
                "excess inventory, high fixed costs, or rapid growth outpacing working capital. "
                "Diagnosis of the root cause is essential before selecting a solution.\n"
                "Recommendation: Immediate actions include invoice factoring, negotiating extended "
                "supplier payment terms, and accelerating receivables collection. Medium-term, "
                "explore a revolving business line of credit with your bank. Prepare a 13-week "
                "cash flow forecast to present to lenders.\n"
                "Risk Level: High"
            )},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "How do I improve my credit score before applying for a mortgage?"},
            {"role": "assistant", "content": (
                "Assessment: Credit scores are primarily driven by payment history, credit "
                "utilisation, length of credit history, and recent enquiries. Mortgage lenders "
                "typically require a minimum score and look for stability over the preceding "
                "12-24 months.\n"
                "Recommendation: Pay all obligations on time, reduce revolving balances to below "
                "30% of limits, avoid opening new credit accounts in the 6 months before application, "
                "and dispute any inaccurate entries on your credit report. Allow at least 6 months "
                "for improvements to be reflected.\n"
                "Risk Level: Low"
            )},
        ]
    },
]


def write_training_file(path: str) -> None:
    """Write training examples to a JSONL file."""
    with open(path, "w") as f:
        for example in TRAINING_EXAMPLES:
            f.write(json.dumps(example) + "\n")
    print(f"✓ Wrote {len(TRAINING_EXAMPLES)} training examples to {path}")


# ---------------------------------------------------------------------------
# STEP 2 — UPLOAD FILE TO OPENAI
# ---------------------------------------------------------------------------

def upload_training_file(path: str) -> str:
    """Upload JSONL file and return the file ID."""
    print(f"\nUploading {path} to OpenAI...")
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    print(f"✓ Uploaded. File ID: {file_id}")
    return file_id


# ---------------------------------------------------------------------------
# STEP 3 — LAUNCH FINE-TUNING JOB
# ---------------------------------------------------------------------------

def create_finetune_job(file_id: str) -> str:
    """Create a fine-tuning job and return the job ID."""
    print(f"\nCreating fine-tuning job on {BASE_MODEL}...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        hyperparameters={
            "n_epochs": N_EPOCHS,
            # Optional: set batch size and learning rate multiplier
            # "batch_size": "auto",
            # "learning_rate_multiplier": "auto",
        },
        # Optional: tag your job for easier identification in the dashboard
        suffix="banking-assistant",   # fine-tuned model name will include this
    )
    job_id = job.id
    print(f"✓ Job created. Job ID: {job_id}")
    print(f"  Monitor at: https://platform.openai.com/finetune/{job_id}")
    return job_id


# ---------------------------------------------------------------------------
# STEP 4 — POLL UNTIL COMPLETE
# ---------------------------------------------------------------------------

def wait_for_job(job_id: str) -> str:
    """Poll job status until succeeded/failed. Returns fine-tuned model name."""
    print(f"\nPolling job {job_id} every {POLL_INTERVAL}s...")
    
    terminal_states = {"succeeded", "failed", "cancelled"}
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        # Show recent events (training loss etc.)
        events = client.fine_tuning.jobs.list_events(job_id, limit=1).data
        latest = events[0].message if events else "—"
        
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status:12s}  |  {latest}")
        
        if status in terminal_states:
            break
        
        time.sleep(POLL_INTERVAL)
    
    if status != "succeeded":
        raise RuntimeError(f"Fine-tuning job {status}. Check the dashboard for details.")
    
    model_id = job.fine_tuned_model
    print(f"\n✓ Fine-tuning complete!")
    print(f"  Fine-tuned model ID: {model_id}")
    return model_id


# ---------------------------------------------------------------------------
# STEP 5 & 6 — COMPARE BASE VS FINE-TUNED MODEL
# ---------------------------------------------------------------------------

def compare_models(finetuned_model_id: str) -> None:
    """Run the same prompt through base and fine-tuned model, print side by side."""
    
    test_questions = [
        "Should I use my savings to buy cryptocurrency?",
        "I missed a credit card payment last month. What should I do?",
    ]
    
    print("\n" + "=" * 70)
    print("  BASE vs FINE-TUNED MODEL COMPARISON")
    print("=" * 70)
    
    for question in test_questions:
        messages_base = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        
        # Base model response
        base_resp = client.chat.completions.create(
            model=BASE_MODEL,
            messages=messages_base,
            max_tokens=300,
            temperature=0,
        )
        base_answer = base_resp.choices[0].message.content.strip()
        
        # Fine-tuned model response
        ft_resp = client.chat.completions.create(
            model=finetuned_model_id,
            messages=messages_base,
            max_tokens=300,
            temperature=0,
        )
        ft_answer = ft_resp.choices[0].message.content.strip()
        
        print(f"\nQuestion: {question}")
        print(f"\n  [BASE MODEL — {BASE_MODEL}]")
        print(f"  {base_answer[:400]}")
        print(f"\n  [FINE-TUNED — {finetuned_model_id}]")
        print(f"  {ft_answer[:400]}")
        print("\n" + "-" * 70)


# ---------------------------------------------------------------------------
# STEP 7 — CLEANUP (OPTIONAL)
# ---------------------------------------------------------------------------

def cleanup(file_id: str, model_id: str) -> None:
    """Delete uploaded file and fine-tuned model to avoid ongoing storage costs."""
    print("\nCleaning up...")
    
    client.files.delete(file_id)
    print(f"  ✓ Deleted training file {file_id}")
    
    # Deleting the fine-tuned model removes it from your account
    # Only do this if you no longer need it
    # client.models.delete(model_id)
    # print(f"  ✓ Deleted fine-tuned model {model_id}")
    
    print("  (Fine-tuned model preserved — delete manually if no longer needed)")
    print(f"  Delete via: client.models.delete('{model_id}')")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(__doc__)
    
    # Step 1: Write training data
    write_training_file(TRAINING_FILE)
    
    # Step 2: Upload to OpenAI
    file_id = upload_training_file(TRAINING_FILE)
    
    # Step 3: Create fine-tuning job
    job_id = create_finetune_job(file_id)
    
    # Step 4: Wait for completion (typically 10-30 mins for small datasets)
    print("\nNote: Fine-tuning typically takes 10-30 minutes for small datasets.")
    print("You can safely Ctrl+C and re-attach later using:")
    print(f"  job = client.fine_tuning.jobs.retrieve('{job_id}')")
    
    finetuned_model_id = wait_for_job(job_id)
    
    # Step 5 & 6: Compare outputs
    compare_models(finetuned_model_id)
    
    # Step 7: Cleanup
    cleanup(file_id, finetuned_model_id)
    
    # Print usage summary
    print("\n" + "=" * 70)
    print("  HOW TO USE YOUR FINE-TUNED MODEL GOING FORWARD")
    print("=" * 70)
    print(f"""
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="{finetuned_model_id}",
        messages=[
            {{"role": "system",  "content": "{SYSTEM_PROMPT[:60]}..."}},
            {{"role": "user",    "content": "Your question here"}}
        ]
    )
    print(response.choices[0].message.content)
    """)


if __name__ == "__main__":
    main()
```
