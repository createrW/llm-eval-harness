# Renting GPUs and hosting for llm-eval-harness: decision memo (2026-10-18)

> **Superseded for deployment (2026-10-23).** Deployment follows the author's hosting plan: the
> evaluations run locally and the dashboard runs on Streamlit Community Cloud ($0). The full local
> grid ran on an Apple M5 laptop. This memo stays as the price reference for an optional rented
> GPU; its setup commands that use plain `uv sync` install the CPU wheel on Linux, so follow the
> README section "Run the grid on a cloud GPU" instead.

**Scope.** This memo covers renting compute for an inference-only eval grid and hosting an always-on Streamlit + MLflow box, on a $0-15 budget. The grid:
- Models: Qwen2.5-0.5B and Llama-3.2-1B.
- HellaSwag: 500 items × 4 endings × 3 prompts.
- GSM8K: 300 items × 3 prompts, greedy decoding, up to 512 new tokens.
- Everything is repeated 3 times.

**Conventions.** Prices are USD per hour, on-demand, for one GPU, read on 2026-10-18 from the linked pages unless marked otherwise. "Research only" means the number comes from the earlier research pass and was not independently re-checked.

## Bottom line

1. **GPU runs:** use a Thunder Compute RTX A6000 (48 GB, $0.35/h, billed per minute) paid by the $20 US-student credit. The full 3-rerun grid takes ~8 GPU-h at batch 1 (~$2.80 of credit) or ~1 h batched, so it costs $0 cash. If Thunder doesn't work out, use RunPod Community RTX A5000/3090 ($0.16/$0.22 per h) or a Vast.ai verified RTX 3090 ($0.17/h), about $1.30-1.80 of compute.
2. **Other $0 routes:**
   - Kaggle notebooks: T4×2/P100, about 30 GPU-h/week, no SSH.
   - Modal L4 serverless: about $1.02/h all-in, inside the $30/month Starter credit.
   - AWS g6.xlarge or GCP g2-standard-4 (L4): paid from sign-up credits, but only once a GPU quota request is approved.
   - Camber via the Student Pack: 5 GPU-h/month, GPU type unknown, so check on signup. Unverified for SSH/Docker. That covers the batched grid (~1-2 h) at $0, with no card.
   - Modal academic program: "up to $10k free compute credits" for "graduate students, labs, and researchers", on application ([modal.com/pricing](https://modal.com/pricing)). Eligibility for a master's project is not confirmed.
3. **Dashboard + MLflow:** host the public dashboard on Streamlit Community Cloud ($0). Run MLflow on an Oracle Always Free A1 VM ($0), or just on the laptop. Fallbacks in order: Azure for Students B1s ($0, no card), then a DigitalOcean 1 GiB Droplet ($6/month).
4. **Expected total:** $0 if the Thunder credit and the Oracle sign-up both work. The worst realistic case is about $11-16: a Vast deposit (minimum not stated in its billing docs) or the $10 RunPod minimum (about $1.50 of it actually used), plus one $6 DigitalOcean month. Either provider fits under $15 if you avoid the DigitalOcean month. Calls to the GPT-4o-mini and Claude Haiku APIs are extra.
5. **Is the laptop enough?** Yes, if you batch generation. On this M5 today, Qwen2.5-0.5B in bf16 decoded 33 tok/s at batch 1 and 566 tok/s at batch 32. That puts the whole grid at about 3-4 h batched, versus about 18 h at batch 1. Rent a GPU for one afternoon only if you want one pinned CUDA environment for all reruns, or if you keep batch-1 decoding.

## Compute need

- **Work.** Each model-run is 6,000 HellaSwag scoring passes (500 × 4 × 3) plus 900 GSM8K generations (300 × 3). The grid is 2 local models × 3 reruns = 6 model-runs: 36,000 scoring passes and 5,400 generations.
- **Tokens.** At a planning average of 300 new tokens per generation, the grid generates 1.62 M tokens. If every answer runs to the 512-token cap, that becomes 2.76 M tokens and about 1.7× the time, so add stop strings. Decoding takes about 90% of the GPU time.
- **Measured on this M5 (2026-10-18).** Setup: Qwen2.5-0.5B base, bf16, MPS, torch 2.14.1 and transformers 5.20.1 from the project venv; 256-token prompt, 128 forced greedy tokens.
  - Decoding: 33 tok/s at batch 1 (30 ms/step), 204 tok/s at batch 8, 566 tok/s at batch 32.
  - Scoring 120-token sequences with a full-vocab log-softmax: 11 seq/s at batch 1, 22 seq/s at batch 16.
  - Script: `scripts/bench_decode.py`.
- **Assumed, not measured.**
  - Llama-3.2-1B decodes at about 25 tok/s at batch 1 on the M5.
  - Batch-1 GPU rates are 35 / 45 / 65 tok/s for T4 / L4-A10 / 3090-class cards. With HF `generate`, models under 2B are bound by kernel-launch overhead, so cards differ less than their memory bandwidth suggests.
  - Batched effective rates, after padding waste, are 300 tok/s on the M5 (Qwen) and 400 / 700 / 1,200 tok/s on the three GPU classes.
  - Each cloud session adds 0.2-0.3 h of HellaSwag and 0.5 h of setup.
  - Re-time with a 20-example GSM8K slice on the first rented box.

| Hardware class | Batch-1 decode | Grid hours, batch 1 | Grid hours, batched (bs 16-32) | Cost: batch 1 → batched |
|---|---|---|---|---|
| T4-class (T4, P100; no bf16, so use fp16) | ~35 tok/s (assumed) | ~14 h | ~2 h | Kaggle $0 ([docs](https://www.kaggle.com/docs/efficient-gpu-usage)); [HF Jobs](https://huggingface.co/docs/hub/jobs-pricing) t4-small $0.40/h → $5.60 → $0.80; AWS g4dn.xlarge $0.526/h ([price map](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json)) → $7.36 → $1.05, covered by credit |
| L4 / A10-class (L4, A10, RTX 4000 Ada) | ~45 tok/s (assumed) | ~11 h | ~1.5 h | [GCP](https://cloud.google.com/products/compute/pricing/accelerator-optimized) g2-standard-4 $0.7068 (medium; see row 4) → $7.78 → $1.06; [AWS](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json) g6.xlarge $0.8048 → $8.85 → $1.21; [HF Jobs](https://huggingface.co/docs/hub/jobs-pricing) l4x1 $0.80 → $8.80 → $1.20; [Modal](https://modal.com/pricing) L4 ≈$1.02 all-in → ≈$11.24 → ≈$1.53 |
| 3090 / 4090 / A5000 / A6000-class | ~65 tok/s (assumed) | ~8 h | ~1 h | [RunPod](https://www.runpod.io/pricing) A5000 $0.16 → $1.28 → $0.16; [Vast](https://console.vast.ai/api/v0/bundles/) 3090 $0.17 → $1.36 → $0.17; RunPod 3090 $0.22 → $1.76 → $0.22; [Thunder](https://www.thundercompute.com/pricing) A6000 $0.35 → $2.80 → $0.35, covered by credit |
| Apple M5 laptop (16 GB, MPS) | 33 tok/s measured (Qwen-0.5B) | ~18 h | ~3-4 h | $0 |

**How the M5 rows add up.** At batch 1: 0.81 M tokens at 33 tok/s plus 0.81 M at ~25 tok/s comes to ≈15.8 h, plus ~2.5 h of scoring. Batched: ≈2.1 h of decoding plus ~1.2 h of scoring.

**Laptop verdict.** Batched, the M5 finishes the grid in an afternoon. If it is a fanless MacBook Air, expect thermal throttling on multi-hour runs (not measured). A cloud GPU buys speed and one pinned CUDA environment for all three reruns. It does not add any capability you lack.

## Ranked GPU VM options

**Sort order.** Rows are sorted by the estimated cash cost of the compute this project uses. Credit-funded options count as $0 and come first; ties are broken by how quickly a new account can actually launch. Minimum deposits are listed under friction.

**Cost column.** "Est. project cost" uses the batch-1 hours from the compute table (T4-class 14 h, L4/A10 11 h, 3090-class 8 h). The batched hours (2 / 1.5 / 1 h) give the figure in [brackets].

| # | Provider | Product / GPU | $/h | Billing | Free credit | Signup friction | Est. project cost | Verified? | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Thunder Compute | RTX A6000 48 GB VM (6 vCPU, 48-64 GB RAM, 100 GB disk included) | 0.35 | Per minute while running. No stop button: snapshot ($0.05/GB-mo), then delete | $20 US-student credit, usable on A6000 and L40 only | Sign up with your usc.edu email; the domain is checked automatically. The page doesn't say whether a card is needed. North America only | ~$2.80 [$0.35] of credit, **$0 cash** | Yes. Pricing page (list prices checked Oct 16, 2026) and students page re-read 2026-10-18. That USC qualifies is an inference | [pricing](https://www.thundercompute.com/pricing), [students](https://www.thundercompute.com/students) |
| 2 | Modal | L4 24 GB serverless function, not a VM | 0.7992 GPU only; ≈1.02 with 2 cores + 16 GiB | Per second; scales to zero. CPU ($0.0000131/core-s) and memory ($0.00000222/GiB-s) billed on top | $30/month (Starter plan). Academic program: up to $10k credits on application (modal.com/pricing); eligibility for a master's project not confirmed | GitHub/Google sign-up, no quota. The page doesn't say whether a card is needed | ≈$11.24 [$1.53] of credit, **$0 cash** | Yes, re-read 2026-10-18. Hourly figure = per-second price × 3600 | [modal.com/pricing](https://modal.com/pricing) |
| 3 | AWS | g6.xlarge, 1× L4 24 GB, 4 vCPU / 16 GiB, us-east-1 | 0.8048 (Spot 0.105) | Per second, 60 s minimum | $100 at sign-up, plus up to $100 for activities | Card and phone. The G/VT vCPU quota defaults to 0: request at least 4 (the Spot quota is separate). The Free plan blocks some services, so upgrade to Paid; remaining credits carry over | $8.85 [$1.21], Spot ≈$1.16, **$0 cash** | On-demand: yes (AWS price feed published 2026-10-15). Spot: yes, re-read 2026-10-18. The official spot feed shows linux g6.xlarge at "0.105" in us-east-1, 0.1039 in us-east-2 and 0.1046 in us-west-2. A reviewer saw third-party trackers (doit.com, not re-checked) showing $0.50-0.60, so the feed and the live console can disagree; check the console before relying on Spot | [price map](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json), [spot feed](https://website.spot.ec2.aws.a2z.com/spot.js), [free tier](https://aws.amazon.com/free/), [FAQ](https://aws.amazon.com/free/free-tier-faqs/) |
| 4 | Google Cloud | g2-standard-4, 1× L4 24 GB, 4 vCPU / 16 GiB, us-central1 | 0.706832 (Spot 0.423016) | Per second, 1 min minimum | $300, valid for 90 days | Card. Upgrade from the Free Trial before any GPU can be attached. The GPUS_ALL_REGIONS quota starts at 0 and must be requested; new billing accounts can be delayed or denied | $7.78 [$1.06], **$0 cash** | Medium. Both official pricing pages (accelerator-optimized, gpus-pricing) came back without a pricing table on 2026-10-18. $0.7068 matches third-party trackers ([holori](https://calculator.holori.com/gcp/vm/g2-standard-4), [economize](https://www.economize.cloud/resources/gcp/pricing/compute-engine/g2-standard-4/)). Not confirmed via the Cloud Billing Catalog API or the calculator. Spot: low (third parties show $0.4028, not $0.423) | [GPU pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [trial terms](https://docs.cloud.google.com/free/docs/free-cloud-features) |
| 5 | RunPod Community | RTX A5000 24 GB Pod | 0.16 | Per second. Volume disk $0.10/GB-mo while running, $0.20 while stopped. No data-transfer fees | Referral bonus only: Google SSO via a referral link, new account, load $10; usually $5 | Card. "As little as $10" to start; prepaid cards need a $100 minimum deposit; credits are non-refundable. Community stock varies | $1.28 [$0.16] | Yes. Page JSON-LD re-read ("Updated October 8, 2026"); the visible price card shows the Secure rate, $0.27 | [pricing](https://www.runpod.io/pricing), [billing](https://docs.runpod.io/references/billing-information), [referrals](https://docs.runpod.io/accounts-billing/referrals) |
| 6 | Vast.ai | RTX 3090 24 GB, verified host, reliability ≥ 0.98 | 0.17 | Per second. Storage (~$0.87/GB-mo on that offer) and bandwidth (~$0.016/GB) are extra; storage bills while stopped | None | Minimum deposit not stated in the billing docs (verify at checkout). A card is required for Stripe purchases; crypto (BitPay, Crypto.com) is accepted. Email verification. Hosts are peer-run | $1.36 plus ~$0.3-0.9 storage [$0.17] | Yes. Live offers API on 2026-10-18: $0.1696, California host, reliability 0.988. Changes hourly | [offers API](https://console.vast.ai/api/v0/bundles/), [billing](https://docs.vast.ai/documentation/reference/billing) |
| 7 | RunPod Community | RTX 3090 24 GB Pod | 0.22 | Per second | Referral only | Same as row 5. One review found Community 3090s sold out | $1.76 [$0.22] | Yes (JSON-LD; Secure is $0.50) | [pricing](https://www.runpod.io/pricing) |
| 8 | RunPod Community | RTX 4090 24 GB Pod | 0.34 | Per second | Referral only | Same as row 5 | ≤$2.72 [$0.34] | Yes (JSON-LD re-read; Secure is $0.89) | [pricing](https://www.runpod.io/pricing) |
| 9 | RunPod Secure | RTX 3090 24 GB Pod | 0.50 | Per second | Referral only | Same as row 5. Default spend limit is $80/h | $4.00 [$0.50] | Yes | [pricing](https://www.runpod.io/pricing) |
| 10 | Hugging Face Jobs | t4-small (T4 16 GB), batch container, no SSH | 0.40 | Per minute, while Starting/Running only | None | Prepaid credit, so a card is required. Default timeout is 30 min | $5.60 [$0.80] | Yes, re-read 2026-10-18. The jobs-pricing page also lists t4-medium $0.60, l4x1 $0.80, per-minute billing and a 30-min default timeout. The same rate appears on the Spaces hardware list | [jobs pricing](https://huggingface.co/docs/hub/jobs-pricing), [HF pricing](https://huggingface.co/pricing) |
| 11 | Lambda Cloud | 1× Quadro RTX 6000 24 GB (Turing, no bf16) | 0.69 | Not stated on the pricing page (column header is "PRICE/GPU/HR"). No stop; terminate only | None | Card; debit and prepaid cards reportedly refused. Capacity varies | $7.59 [$1.04] | Price: yes, re-read 2026-10-18. Billing granularity unsourced | [lambda.ai/pricing](https://lambda.ai/pricing) |
| 12 | DigitalOcean | GPU Droplet, RTX 4000 Ada 20 GB | 0.76 | Per second, 5-minute minimum for GPU Droplets (pricing page). The 60 s / $0.01 minimum applies to Basic Droplets ([droplets page](https://www.digitalocean.com/pricing/droplets)) | The $5 promo credit is only a worked example on the page | Card. Powered-off Droplets keep billing | $8.36 [$1.14] | Yes, re-read 2026-10-18 ("$0.76/GPU/hour") | [gpu-droplets](https://www.digitalocean.com/pricing/gpu-droplets) |
| 13 | Hugging Face Jobs | l4x1 (L4 24 GB, 8 vCPU, 30 GB RAM) | 0.80 | Per minute | None | Same as row 10 | $8.80 [$1.20] | Yes | [jobs pricing](https://huggingface.co/docs/hub/jobs-pricing) |
| 14 | Lambda Cloud | 1× A10 24 GB | 1.29 | Not stated on the pricing page; terminate only | None | Card | $14.19 [$1.94] | Price: yes, re-read 2026-10-18. Billing granularity unsourced | [lambda.ai/pricing](https://lambda.ai/pricing) |

**Checked but not recommended:**
- Vast verified L4 $0.322 and A10 $0.243 ([API](https://console.vast.ai/api/v0/bundles/)): very few offers.
- RunPod Community L4 $0.44 and A6000 $0.33 ([pricing](https://www.runpod.io/pricing)).
- AWS g4dn.xlarge T4 $0.526 ([price map](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json)) and GCP n1-standard-4 + T4 $0.54 ([general-purpose](https://cloud.google.com/products/compute/pricing/general-purpose) plus the accelerator page): no bf16, and about the same price as an L4.
- Azure NC4as_T4_v3 $0.526 ([retail API](https://prices.azure.com/api/retail/prices?$filter=serviceName%20eq%20'Virtual%20Machines'%20and%20armRegionName%20eq%20'eastus'%20and%20armSkuName%20eq%20'Standard_NC4as_T4_v3')): not possible on Azure for Students; needs pay-as-you-go plus a quota request.
- OCI VM.GPU.A10.1 $2.00 ([price list](https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/?currencyCode=USD&partNumber=B95909)): the A10 limit is "Contact Us".
- Shadeform L4 $0.95 ([compare](https://shadeform.com/directory/gpus/compare/A100/L4)).
- Hyperbolic: H100 and larger only, from $3.19 ([marketplace](https://www.hyperbolic.ai/marketplace)).
- TensorDock: its price table is dated 2024-07-24 and its APIs returned nothing ([page](https://www.tensordock.com/cloud-gpus.html)).
- Crusoe L40S $1.50 ([pricing](https://www.crusoe.ai/cloud/pricing)).
- Nebius L40S $1.55, with a $25 minimum first payment and the free trial suspended ([prices](https://nebius.com/prices), [trial](https://docs.nebius.com/signup-billing/free-trial)).
- SaladCloud RTX 3090 24 GB: $0.17/h at High priority, $0.143 Medium, $0.117 Low, and $0.09/h at Lowest (interruptible). RTX 4090 runs from $0.33 (High) down to $0.16 (Lowest). vCPU and RAM are included, billing is per second, and there is no minimum commitment. It is container-only on consumer hosts, and the page does not say whether SSH is available. Skip it for the test-retest reruns (node and hardware variance); it is fine for smoke tests ([salad.com/pricing](https://salad.com/pricing)).

### Top-5: first steps

**1. Thunder Compute A6000 (student credit).**
- Sign up with your usc.edu address and check that the $20 credit appears; the page says it is added automatically.
- Create a 1× RTX A6000 instance from the console, CLI, or VS Code extension, then SSH in.
- CUDA 13.0, driver 580, PyTorch 2.9.0+cu128, and JupyterLab come preinstalled. The docs say "Do not attempt to reinstall CUDA", so `uv sync` the project's pinned torch into a venv.
- Time a 20-example GSM8K slice first. If tok/s is far below the RunPod/Vast expectation, switch providers. The [compatibility page](https://www.thundercompute.com/docs/compatibility) lists no PyTorch issues; its known gaps are UVM, CUDA MPS, FFmpeg GPU, and GPU profilers.
- When you finish, snapshot if you want to resume, then delete the instance. There is no stop and no documented auto-stop.

**2. Modal L4 (serverless, $30/month credit).**
- Run `pip install modal` and `modal setup`.
- Wrap the runner in `@app.function(gpu="L4", timeout=...)`, using an image built from `requirements.txt`. Set the timeout explicitly.
- Store `HF_TOKEN` as a Modal Secret, write parquet to a `modal.Volume`, and pull it with `modal volume get`.
- Start the grid with `modal run`. Containers scale to zero when idle, so a forgotten shutdown costs nothing.
- The batch-1 grid uses about $11 of the $30 credit.
- Academic program: up to $10k credits on application ([modal.com/pricing](https://modal.com/pricing)). Eligibility for a master's project is not confirmed.

**3. AWS g6.xlarge (L4, sign-up credits).**
- Create the account and upgrade to the Paid plan. The Free plan blocks some services; remaining credits carry over.
- In Service Quotas → EC2, request "Running On-Demand G and VT instances" = 8 vCPUs in us-east-1. Also request "All G and VT Spot Instance Requests" if you will use Spot.
- Launch g6.xlarge with an AWS Deep Learning AMI (GPU, PyTorch), `uv sync`, and run under tmux.
- Set AWS Budgets alerts at $5 and $15.
- Terminate the instance when done; stopping it is not enough, because EBS keeps billing.

**4. Google Cloud g2-standard-4 (L4, $300 trial).**
- Create the account, then upgrade the billing account to Paid. Free Trial accounts can't attach GPUs or request quota, and the credit still expires 90 days after sign-up.
- Request GPUS_ALL_REGIONS ≥ 1 (and NVIDIA_L4_GPUS in us-central1 if prompted). Expect a wait of up to a day, or a refusal for a new account.
- Launch g2-standard-4 from a Deep Learning VM image (PyTorch/CUDA).
- Set budget alerts at $5 and $15. Delete the VM and its disk afterwards.

**5. RunPod Community A5000/3090 (cheapest paid).**
- Sign up through a referral link with Google SSO if you want the bonus, and load $10. Credits are non-refundable.
- Deploy a GPU Pod → Community Cloud → RTX A5000 ($0.16) or RTX 3090 ($0.22), using the official RunPod PyTorch template (JupyterLab on port 8888, SSH over exposed TCP). Add your SSH key.
- If Community stock is gone, take Secure (3090 at $0.50).
- Copy results off after every run, then terminate the Pod.
- **Vast.ai works the same way:** filter to verified hosts with reliability ≥ 0.98 and an on-demand RTX 3090 (about $0.17), use the PyTorch template, add a small deposit (the minimum is not stated in the billing docs, so check at checkout), and destroy the instance afterwards, because storage bills while it is stopped.

## Free and notebook tiers

| Option | GPU | Free allowance and limits | SSH / background | Verified? | Source |
|---|---|---|---|---|---|
| Kaggle Notebooks | P100 16 GB, or 2× T4 16 GB (two separate cards) | About 30 GPU-h/week, "sometimes higher". 2×T4 uses 1 h of quota per runtime hour. Sessions 9-12 h (sources disagree). Phone verification reportedly needed for GPU and internet | No SSH. "Save Version / Run All" runs in the background | Medium. The docs are JS-rendered, so figures come from kaggle.com search snippets. A Sept-2026 community thread (unverified) reports the P100 retired and GPU availability getting worse | [efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [T4 announcement](https://www.kaggle.com/product-feedback/361104), [feedback thread](https://www.kaggle.com/discussions/product-feedback/743898) |
| Google Colab, free | Varies; no type guaranteed | Sessions "at most 12 hours", idle timeouts, limits not published | No background execution. SSH and remote desktop are not allowed on free runtimes without compute units | Yes (FAQ) | [Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| Colab Pro for Education: CLOSED to new signups (per FAQ, 2026-10-18); existing subscribers only | T4/L4/A100 with priority (paid Pro) | FAQ: "They are no longer available for new signups at this time." Paid Colab Pro is still $9.99/mo, but the signup page rendered without prices for me, so that figure is low confidence. A Google AI Pro student offer that may include Colab benefits is reported only by third parties (unverified) | Background runs need Pro+ | Yes (FAQ) | [Colab FAQ](https://research.google.com/colaboratory/faq.html), [signup](https://colab.research.google.com/signup) |
| HF ZeroGPU | Half an RTX Pro 6000 (48 GB) | 5 GPU-min/day free (40 with PRO). Gradio only. Free accounts get up to 2 Spaces and need a verified email and an account older than 30 days | No | Research only (high) | [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu), [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) |
| GitHub Codespaces (Education) | None; CPU only, GPU machine type deprecated | Inferred from Pro (180 h compute, 20 GB). The billing doc lists Pro at "180 hrs" of compute and 20 GB-month, with a usage multiplier of 2 on a 2-core machine (about 90 h on 2 cores), and does not mention the Education plan. 30-min idle timeout, 12 h maximum lifetime | SSH: yes | Partly (Pro allowance yes; Education allowance inferred) | [students](https://github.com/education/students), [billing](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces) |
| Camber (Student Pack) | Not stated | Pack lists "40 CPU hours, 5 GPU hours, 50 GB of storage, and 50 agent messages per month", per month (Pack page, 2026-10-18), "while you're enrolled". 5 GPU-h covers the batched grid (~1-2 h) at $0, with no card | Unverified for SSH/Docker. The Pack mentions "long jobs keep running after you close your laptop" | Allowance yes (Pack re-read 2026-10-18). GPU type, SSH and Docker unverified. An older announcement lists different allowances | [Pack](https://education.github.com/pack) |
| Lightning AI Studios | T4/L4 (reported) | Down since 2026-10-16 for an unresolved security incident (status.lightning.ai: "Service is down to address a security incident"; Investigating 03:08 UTC, "We are currently doing an emergency maintenance on the service"). The pricing and blog pages return 503. Skip this week | n/a | Status page: yes (re-read 2026-10-18). Incident details unknown, because the linked blog post returns 503 | [status.lightning.ai](https://status.lightning.ai/), [incident blog (503)](https://lightning.ai/blog/oct-2026-incident), [lightning.ai/pricing (503)](https://lightning.ai/pricing) |

**Enough when:**
- You run the full grid on Kaggle as background commits. Batched it is about 2 h on a T4; at batch 1 it is about 14 h. Either fits inside about 30 h/week.
- You use Colab free for Day 1-2 smoke tests.

**Bites when:**
- You need SSH or your Docker image. Neither Kaggle nor Colab offers either.
- You need bf16. T4 and P100 don't support it, so you fall back to fp16, which shifts log-likelihoods relative to the bf16 runs on the M5.
- A session cap kills a long run. Checkpoint each grid cell and write parquet incrementally.
- The assigned GPU changes between sessions (P100 vs T4). That breaks the pinned-hardware assumption behind the test-retest claim.
- You use Llama-3.2-1B. It is gated, so `HF_TOKEN` must be in Kaggle Secrets or Colab userdata.

## Always-on host for Streamlit + MLflow

| Option | Spec | Price | Can run | Catches | Verified? | Source |
|---|---|---|---|---|---|---|
| Streamlit Community Cloud | 0.078-2 cores, 690 MB-2.7 GB RAM, up to 50 GB (limits "as of February 2024") | $0 ("Totally free") | Dashboard only, reading committed parquet files | Sleeps after 12 h without traffic; one click wakes it. Cannot host the MLflow server | Yes | [manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [streamlit.io/cloud](https://streamlit.io/cloud) |
| Oracle Always Free, VM.Standard.A1.Flex (Arm) | 2 OCPU + 12 GB total; 200 GB block storage; 10 TB/month egress | $0 | Both, comfortably (docker compose: MLflow + SQLite/Postgres + Streamlit) | Card and mobile number at sign-up. Home region is fixed at sign-up. A1 capacity often unavailable. Instances are reclaimed after 7 idle days (p95 CPU, network, and A1 memory all under 20%). Needs arm64 images | Yes, re-read 2026-10-18 | [Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [Free Tier](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm) |
| Oracle Always Free, VM.Standard.E2.1.Micro (up to 2) | 1/8 OCPU (can burst), 1 GB, 50 Mbps | $0 | MLflow only, with swap | Same rules as A1 | Yes | [Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) |
| Azure for Students, B1s | 1 vCPU / 1 GiB | $0: 750 h/month each of B1s, B2pts v2 and B2ats v2 for 12 months, plus $100 credit | MLflow only, with swap | No card; school email. Disabled when the credit or the 12 months run out, unless you move to pay-as-you-go | Yes | [Azure for Students](https://azure.microsoft.com/en-us/free/students/) |
| GCP e2-micro Free Tier | 2 shared vCPU / 1 GB, 30 GB-month disk | $0 in us-west1, us-central1 or us-east1 | MLflow only, with swap | Card required. 1 GB/month egress. An external IPv4 fee is reported but unverified | Yes | [free cloud features](https://docs.cloud.google.com/free/docs/free-cloud-features) |
| DigitalOcean Basic Droplet, 1 GiB | 1 vCPU / 1 GiB / 25 GiB | $6/month ($0.00893/h) | Both, but tight; add swap | Powered-off Droplets still bill. The Student Pack credit ended 2026-08-01 | Yes | [droplets](https://www.digitalocean.com/pricing/droplets) |
| DigitalOcean Basic Droplet, 2 GiB | 1 vCPU / 2 GiB / 50 GiB | $12/month ($0.01786/h) | Both, comfortably | Uses most of the budget each month | Yes, re-read 2026-10-18 | [droplets](https://www.digitalocean.com/pricing/droplets) |
| Heroku Basic dyno + Postgres Essential-0 | 0.5 GB RAM | $7 + $5 per month; the Pack lists a $13/month credit for 24 months | MLflow (Postgres backend; artifacts need external storage) | Dyno disk is ephemeral. Eco dynos ($5) sleep after 30 min. The credit appears only on the Pack page, not Heroku's | Partly | [heroku.com/pricing](https://www.heroku.com/pricing), [Pack](https://education.github.com/pack) |
| Hugging Face Spaces, CPU Basic | 2 vCPU / 16 GB / 50 GB, not persistent | Hardware is free, but Docker Spaces need PRO ($9/month) | Dashboard only | Sleeps when unused; disk is not persistent | Yes | [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview), [pricing](https://huggingface.co/pricing) |
| Fly.io shared-cpu-1x, 1 GB | 1 GB | $6.70/month (iad) | Both | No free tier for new organizations. The trial needs no card; a card is required after 2 h of Machine runtime or 7 days | Yes | [Fly pricing](https://docs.fly.io/about/pricing) |
| Render, free web service | 0.1 CPU / 512 MB | $0 (750 instance-hours/month) | Dashboard only | Spins down after 15 min idle. Ephemeral filesystem, no persistent disks | Research only (high) | [Render free](https://render.com/docs/free) |
| Railway Hobby | Usage-based | $5/month, including $5 of usage | Both | About $7/month net for 1 GB always-on (estimate). The Free plan gives only $1/month of credit | Research only (high) | [Railway pricing](https://railway.com/pricing) |

**Not available this week:** Hetzner and Koyeb.
- Hetzner: on [regular-performance](https://www.hetzner.com/cloud/regular-performance/), every CPX plan's Create button shows "This product is currently unavailable. Please check back later.", and [hetzner.com/cloud](https://www.hetzner.com/cloud) marks Cost-Optimized as "Currently not available". US locations (ASH/HIL) offer only CPX11-51. If stock returns, CPX11 (2 vCPU / 2 GB) would be a candidate always-on box, so re-check before falling back to DigitalOcean.
- Koyeb ([new organizations need Pro at $29/month](https://www.koyeb.com/pricing)).

**Recommendation.**
- **Dashboard:** put the public dashboard on Streamlit Community Cloud, reading result parquet files committed to the repo. Recruiters get a stable `*.streamlit.app` link at $0.
- **MLflow:** run it on an Oracle Always Free A1 VM with docker compose (MLflow, SQLite, local artifacts, basic auth in front).
  - A small shape such as 1 OCPU / 4 GB should keep normal memory use above Oracle's 20% idle threshold. That is my reading of the reclaim rule, not an Oracle guarantee.
  - Even simpler: during GPU runs, log to a local `mlruns/` or SQLite file on the GPU box, rsync it home, and serve `mlflow ui` from the laptop or the Oracle VM.
- **Fallbacks:** if Oracle rejects your card or has no A1 capacity, use Azure for Students B1s ($0, no card), then a DigitalOcean 1 GiB Droplet ($6/month).

## Fact-check of the 2026-10-17 hosting plan

| Claim (from memory, 2026-10-17) | Verdict | Current fact (2026-10-18) | Source |
|---|---|---|---|
| Oracle Always Free: A1 4 OCPU + 24 GB, 200 GB block, 10 TB/mo egress, permanently free, card required | OUTDATED (A1 size); the rest CONFIRMED | A1 is now 2 OCPU + 12 GB in total (1,500 OCPU-h + 9,000 GB-h/month). 200 GB block (boot and block combined) and 10 TB/month outbound are confirmed. Always Free offers "never expire", but idle instances can be reclaimed and everything must live in the home region. "Most users" need a card and a mobile number; the card isn't charged unless you upgrade. If A1 usage exceeds 2/12 when the trial ends, all A1 instances are disabled, then deleted after 30 days | [Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [Free Tier](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm) |
| AWS Free Tier: 750 h/mo t2/t3.micro for 12 months, RDS db.t3.micro 750 h, S3 5 GB | OUTDATED for new accounts | New accounts get $100 at sign-up plus up to $100 for activities. The Free plan ends after 6 months or when credits run out. Free-plan accounts have only Always Free offers and can't use some services. Upgrading to Paid keeps remaining credits; the FAQ gives both 6 and 12 months as the credit expiry. The 750-hour offers are not on the current pages; they belong to accounts opened before 2025-07-15 (date from an AWS docs search snippet) | [aws.amazon.com/free](https://aws.amazon.com/free/), [AWS Billing docs](https://docs.aws.amazon.com/en_us/awsaccountbilling/latest/aboutv2/free-tier.html), [FAQ](https://aws.amazon.com/free/free-tier-faqs/) |
| GitHub Student Pack: $100 AWS, $200 DigitalOcean, $50 Azure credit | WRONG / OUTDATED | The Pack page lists no AWS or DigitalOcean credit. The DO offer ended: last redemption 2026-07-31, credits expired 2026-08-01. Azure is $100 (age 18+), not $50. Still useful: Heroku $13/month for 24 months, Codespaces Pro, and the Camber student plan (5 GPU-h/month). Also lists VS Dev Essentials $200 Azure credit (1st month). Azure GPU VMs still need pay-as-you-go plus a quota request | [Pack](https://education.github.com/pack), [DO end notice](https://github.com/orgs/community/discussions/201240) |
| Fly.io free tier: 3× shared-cpu-1x 256 MB, 3 GB Postgres, 160 GB outbound | OUTDATED | New organizations get no free tier and no monthly allowance. The trial is 2 h of Machine runtime or 7 days, whichever comes first. It needs no card ("You don't need a credit card to start the trial"); a card is required only to keep going after it ends. Old allowances survive only for orgs on plans discontinued in October 2024. Prices: 256 MB $2.19/month, 1 GB $6.70/month (iad); Managed Postgres from $38/month | [Fly pricing](https://docs.fly.io/about/pricing) |
| Supabase free tier: Postgres 500 MB, 2 projects | CONFIRMED | $0 plan: 500 MB database, 2 active projects, 5 GB egress, 1 GB file storage, 50,000 MAU. Free projects pause after 1 week of inactivity | [supabase.com/pricing](https://supabase.com/pricing) |
| Streamlit Community Cloud: unlimited public apps, 1 GB RAM, free | Free CONFIRMED; RAM OUTDATED; "unlimited" UNVERIFIED | "Totally free". Limits as of February 2024 ("may change"): 0.078-2 cores, 690 MB-2.7 GB memory, up to 50 GB storage. Apps sleep after 12 h without traffic. None of the pages I read states an app-count limit or says "unlimited" | [manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [streamlit.io/cloud](https://streamlit.io/cloud) |
| Hugging Face Spaces: unlimited free CPU Spaces; GPU T4 $0.60/h | OUTDATED | Only Static Spaces are free. Creating Gradio or Docker Spaces (Streamlit runs as Docker) now needs PRO at $9/month; free accounts may host up to 2 ZeroGPU Gradio Spaces. T4 small is $0.40/h, T4 medium $0.60/h, L4 $0.80/h | [Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview), [HF pricing](https://huggingface.co/pricing) |
| Modal: $30/month free credit with no card; A10 $0.60/h; A100 40 GB $1.50/h | WRONG (GPU prices); no-card claim UNVERIFIED | The $30/month Starter credit is confirmed. A10 is $0.000306/s, about $1.10/h; A100 40 GB is $0.000583/s, about $2.10/h. Both are GPU-only, with CPU and memory billed on top. The pricing page doesn't say whether a card is needed | [modal.com/pricing](https://modal.com/pricing) |
| Google Cloud: $300 credit valid for 90 days | CONFIRMED | $300 Welcome credit for 90 days. A Free Trial account can't attach GPUs or request quota. After you upgrade, leftover credit still expires on day 90 | [free cloud features](https://docs.cloud.google.com/free/docs/free-cloud-features) |
| Azure: $200 credit for 30 days, plus $100 Azure for Students | CONFIRMED (two separate offers) | Free account: $200 for 30 days. It needs a non-prepaid credit or debit card and a phone number, and the credit can't be used on Spot VMs or Marketplace. Azure for Students: $100 for 12 months, no card, full-time students, renewable yearly. Both include 750 h/month of B1s, B2pts v2 and B2ats v2 for 12 months. Student subscriptions can't run GPU VMs (3-vCPU cap per a Microsoft Q&A answer; not re-verified) | [Azure free account](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account), [Azure for Students](https://azure.microsoft.com/en-us/free/students/) |
| RunPod: RTX 4090 spot/community around $0.40/h | CONFIRMED (approximately) | Community on-demand is $0.34/h and Secure is $0.89/h (page updated 2026-10-08). The pricing page shows no spot price. For this project the A5000 ($0.16) and 3090 ($0.22) are cheaper and enough | [runpod.io/pricing](https://www.runpod.io/pricing) |
| Render.com: free web service that sleeps after 15 min idle | CONFIRMED | Spins down after 15 min without inbound traffic and takes about a minute to wake. 750 free instance-hours/month. Ephemeral filesystem, no persistent disks, and free Postgres expires after 30 days, so it is unfit for MLflow state | [Render free](https://render.com/docs/free) |
| Railway: $5/month starter | OUTDATED (plan name and terms) | Hobby is $5/month and includes $5 of usage. New users get a $5, 30-day trial with no card. After that, the Free plan gives $1/month of credit (up to 1 vCPU / 0.5 GB per service), which can't keep a service always on | [Railway pricing](https://railway.com/pricing) |

## Pitfalls

**Do the slow sign-ups on day 1:**
- Thunder (usc.edu email).
- Oracle: card and phone; the home region is chosen once and can't be changed.
- Kaggle phone verification.
- Meta's license for Llama-3.2-1B, plus an `HF_TOKEN` on every GPU box.
- AWS and GCP only if you want the hyperscaler route. AWS needs a G/VT quota request (default 0; Spot quota separate). GCP needs a paid upgrade and then GPUS_ALL_REGIONS. Approval can take hours to days, or be refused for new accounts. The OCI A10 limit is "Contact Us".

**Billing alerts and caps:**
- Set AWS Budgets or GCP budget alerts at $5 and $15.
- Thunder emails you about running instances and has a credit-balance alert API ([billing](https://www.thundercompute.com/docs/billing.md)).
- RunPod's default spend limit is $80/h; RunPod and Vast are prepaid, so a small deposit caps your loss. RunPod's minimum is $10 ($100 per transaction for prepaid cards); Vast's minimum isn't stated in its docs.

**Stopping is not always stopping:**
- Thunder has no stop: snapshot, then delete ([guide](https://www.thundercompute.com/docs/guides/stopping-instances.md)).
- Lambda is terminate-only.
- DigitalOcean bills powered-off Droplets.
- RunPod bills stopped volume disk at $0.20/GB-mo. When your balance hits $0 it terminates Pods that have no network volume, and the data can't be recovered.
- Vast storage bills while the instance is stopped.
- Stopped AWS and GCP VMs still bill for disks.
- Copy the parquet files and the MLflow export off the box after every run, then delete it.

**Spot and interruptible instances:**
- AWS Spot gives a 2-minute warning. The official spot feed showed $0.105/h for g6.xlarge in us-east-1 on 2026-10-18; confirm in the console, since trackers disagree.
- Vast interruptible instances pause when outbid, with no documented notice.
- Use on-demand for the three test-retest reruns. Checkpoint per (model, task, prompt, seed) cell and append to parquet as you go.

**Egress is negligible here; results are a few MB:**
- Thunder charges $0 for egress, and RunPod has no transfer fees.
- Vast charges about $0.016/GB on the verified offer.
- GCP's Free Tier includes 1 GB/month of egress from North America.
- Download model weights on the GPU box; don't upload them from the laptop.

**Idle hosts sleep or get reclaimed:**
- Oracle reclaims Always Free VMs that stay idle for 7 days.
- Streamlit Cloud apps sleep after 12 h without traffic, Render after 15 min, and Codespaces after a 30-min idle timeout.
- Keep the README's screenshots current so a sleeping app never blocks a reviewer.

**Credit expiry and revocation:**
- GCP's $300 expires on day 90 even after you upgrade.
- AWS closes the Free plan at 6 months; the FAQ gives both 6 and 12 months for credit expiry.
- Azure's free $200 lasts 30 days; Azure for Students lasts 12 months.
- Oracle's $300 trial lasts 30 days, and paid resources created with it are reclaimed.
- Modal's $30 is per month; rollover is not stated.
- Thunder's promotional credit "can be revoked".
- RunPod and Thunder preloaded credit is non-refundable. Vast says "In most cases we can refund unspent credits", on request through website chat; credit bought with crypto is not refunded.

**Reproducibility, which the "test-retest" claim depends on:**
- Run all three reruns on one GPU model, Docker image, dtype and batch size; padding and batch shape change the numerics too.
- T4, P100 and Turing cards (Quadro RTX 6000) have no bf16, so use fp16 there.
- Log GPU, driver, CUDA and torch versions to MLflow.
- Never mix M5 and CUDA runs within one rerun set.
- Thunder ships CUDA 13.0, driver 580 and PyTorch 2.9.0: install the project's own torch in a venv and leave CUDA alone ([specs](https://www.thundercompute.com/docs/technical-specs.md)).

**Timeouts:**
- HF Jobs defaults to 30 min, so pass `--timeout`.
- Modal functions need an explicit `timeout`.
- Kaggle and Colab sessions cap at roughly 9-12 h.
- On SSH boxes, run under `tmux` or `nohup`.

**Skip this week:**
- Lightning AI: down since 2026-10-16 for an unresolved security incident ([status.lightning.ai](https://status.lightning.ai/)); the pricing and blog pages return 503. As a precaution (not a reported fact), rotate any keys you stored there.
- Hetzner: every CPX plan shows "currently unavailable" ([regular-performance](https://www.hetzner.com/cloud/regular-performance/)). Re-check CPX11 before falling back to DigitalOcean.
- Koyeb: Pro required at $29/month.
- Nebius: trial suspended, $25 minimum.
- Hyperbolic: H100 and up only.

## Sources

- https://www.thundercompute.com/pricing
- https://www.thundercompute.com/students
- https://www.thundercompute.com/docs/billing.md
- https://www.thundercompute.com/docs/guides/stopping-instances.md
- https://www.thundercompute.com/docs/technical-specs.md
- https://www.thundercompute.com/docs/compatibility
- https://modal.com/pricing
- https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json
- https://website.spot.ec2.aws.a2z.com/spot.js
- https://aws.amazon.com/free/
- https://aws.amazon.com/free/free-tier-faqs/
- https://docs.aws.amazon.com/en_us/awsaccountbilling/latest/aboutv2/free-tier.html
- https://cloud.google.com/products/compute/pricing/accelerator-optimized
- https://cloud.google.com/products/compute/pricing/general-purpose
- https://docs.cloud.google.com/free/docs/free-cloud-features
- https://www.runpod.io/pricing
- https://docs.runpod.io/references/billing-information
- https://docs.runpod.io/accounts-billing/referrals
- https://console.vast.ai/api/v0/bundles/
- https://docs.vast.ai/documentation/reference/billing
- https://huggingface.co/docs/hub/jobs-pricing
- https://huggingface.co/pricing
- https://huggingface.co/docs/hub/en/spaces-overview
- https://huggingface.co/docs/hub/en/spaces-zerogpu
- https://lambda.ai/pricing
- https://www.digitalocean.com/pricing/gpu-droplets
- https://www.digitalocean.com/pricing/droplets
- https://azure.microsoft.com/en-us/free/students/
- https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account
- https://prices.azure.com/api/retail/prices?$filter=serviceName%20eq%20'Virtual%20Machines'%20and%20armRegionName%20eq%20'eastus'%20and%20armSkuName%20eq%20'Standard_NC4as_T4_v3'
- https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
- https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier.htm
- https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/?currencyCode=USD&partNumber=B95909
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://www.kaggle.com/product-feedback/361104
- https://www.kaggle.com/discussions/product-feedback/743898
- https://research.google.com/colaboratory/faq.html
- https://colab.research.google.com/signup
- https://education.github.com/pack
- https://github.com/orgs/community/discussions/201240
- https://github.com/education/students
- https://docs.github.com/en/billing/concepts/product-billing/github-codespaces
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- https://streamlit.io/cloud
- https://www.heroku.com/pricing
- https://docs.fly.io/about/pricing
- https://render.com/docs/free
- https://railway.com/pricing
- https://supabase.com/pricing
- https://lightning.ai/pricing (HTTP 503 on 2026-10-18)
- https://status.lightning.ai/
- https://lightning.ai/blog/oct-2026-incident (HTTP 503 on 2026-10-18)
- https://salad.com/pricing
- https://calculator.holori.com/gcp/vm/g2-standard-4 (third-party, GCP cross-check)
- https://www.economize.cloud/resources/gcp/pricing/compute-engine/g2-standard-4/ (third-party, GCP cross-check)
- https://shadeform.com/directory/gpus/compare/A100/L4
- https://www.hyperbolic.ai/marketplace
- https://www.tensordock.com/cloud-gpus.html
- https://www.crusoe.ai/cloud/pricing
- https://nebius.com/prices
- https://docs.nebius.com/signup-billing/free-trial
- https://www.hetzner.com/cloud/regular-performance/
- https://www.hetzner.com/cloud
- https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/ (15 June 2026 price increase only; not the availability source)
- https://www.koyeb.com/pricing
