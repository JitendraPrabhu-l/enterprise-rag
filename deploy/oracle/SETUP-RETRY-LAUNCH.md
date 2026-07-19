# Automated retry for Oracle Cloud A1.Flex capacity

The Oracle Cloud console's "Out of capacity for shape VM.Standard.A1.Flex"
error is not something you did wrong — the Always Free ARM pool is
genuinely oversubscribed in several regions (Hyderabad among them), and
capacity opens up unpredictably as other tenants' instances terminate.
Manually re-clicking Create every so often works, but `retry-launch.ps1`
does the same thing unattended: it retries your exact 4 OCPU / 24 GB
launch request on an interval until it succeeds, then prints the new
instance's public IP.

This is a one-time setup (~15 minutes), then the script just runs.

## 1. Install the OCI CLI

PowerShell (as your normal user, not elevated):

```powershell
irm https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.ps1 | iex
```

The installer will ask a few yes/no questions (accept defaults) and may
prompt you to add itself to PATH — accept that too. Close and reopen your
terminal afterward so PATH picks it up, then confirm:

```powershell
oci --version
```

## 2. Authenticate the CLI to your account

```powershell
oci setup config
```

This asks a sequence of questions:

- **User OCID** — Console → profile icon (top right) → **My profile** →
  copy the OCID shown near the top (`ocid1.user.oc1..`).
- **Tenancy OCID** — same profile page, left sidebar has "Tenancy:
  <name>" → click it → copy that OCID (`ocid1.tenancy.oc1..`).
- **Region** — e.g. `ap-hyderabad-1` (whatever you picked when creating
  the VM earlier).
- **Generate a new API signing key pair?** → **Y** (let it generate one;
  do not reuse your SSH key for this — it's a separate credential).
- Accept the default save location it suggests (`~/.oci/`).

The setup prints a **public key block** at the end and tells you to
upload it. Do that now:

Console → profile icon → **My profile** → **API keys** (left sidebar) →
**Add API key** → **Paste a public key** → paste the block the CLI
printed → **Add**.

Verify the CLI can now reach your account:

```powershell
oci iam region list
```

A JSON list of regions means you're authenticated. Any error here means
`retry-launch.ps1` won't be able to authenticate either — fix this step
before continuing.

## 3. Gather the OCIDs the script needs

Open `retry-launch.config.json` in this folder and fill in each value.
Where to find each one:

| Field | Where to find it |
|---|---|
| `compartment_id` | Use your **tenancy OCID** here (from step 2) unless you deliberately created a sub-compartment — most personal accounts launch everything directly in the root/tenancy compartment. |
| `availability_domain` | Console → **Compute → Instances → Create Instance** (don't submit — just open the wizard) → the Availability Domain dropdown shows the exact string, e.g. `wIqA:AP-HYDERABAD-1-AD-1`. Copy it exactly, including the `wIqA:` prefix if present — this is tenancy-specific and not just "AD-1". |
| `subnet_id` | Console → **Networking → Virtual Cloud Networks** → your VCN → **Subnets** → click the public subnet → copy its OCID from the subnet's detail page. |
| `image_id` | Run `oci compute image list --compartment-id <your-tenancy-ocid> --operating-system "Canonical Ubuntu" --operating-system-version "24.04" --shape "VM.Standard.A1.Flex" --query "data[0].id" --raw-output` — this returns the correct ARM-compatible Ubuntu 24.04 image OCID for your region directly. |
| `ssh_public_key_path` | If you don't already have a key pair for this: `ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\oracle_rag_key"` (no passphrase needed, or add one if you prefer). Point this field at the resulting `.pub` file. |

Leave `shape`, `ocpus`, `memory_in_gbs`, `boot_volume_size_in_gbs` as
they are — they already match the Always-Free config you were trying to
launch in the console (4 OCPU / 24 GB / 150 GB boot volume).

## 4. Run it

```powershell
cd D:\RAG\deploy\oracle
.\retry-launch.ps1
```

It will:
1. Verify the OCI CLI is installed and authenticated.
2. Attempt the launch.
3. On "Out of capacity" or a transient network error (connection
   timeout/reset), wait the configured interval (default 120s via
   `retry_interval_seconds`) and try again.
4. On Oracle's own API rate limit (429 TooManyRequests — this happens
   naturally after dozens of attempts over an hour or more), back off
   much longer (10x the normal interval, minimum 10 minutes) before
   resuming, so the loop respects the rate limit instead of hitting it
   again immediately.
5. Repeats 3-4 indefinitely until it succeeds or you press **Ctrl+C**.
6. On success, print the instance OCID, its public IP, and the exact
   `ssh` command to connect.
7. On any OTHER error (bad OCID, auth failure, quota exceeded), it stops
   immediately with the real error message — those need a fix, not a
   retry.

Every attempt's full output (not just what fits in the terminal) is
also appended to `retry-launch.log` in this folder, so you can always
check exactly what Oracle returned without re-running anything by hand.

Leave the terminal window open and your PC awake/unsleeping while it
runs — closing the terminal or letting the machine sleep stops the loop
(it's a foreground PowerShell process, not a scheduled/background
service). Early morning IST (roughly 5–8 AM) has historically been the
best-odds window for Hyderabad A1 capacity, but the script works fine
left running for hours regardless.

## 5. After it succeeds

Add the two ingress rules (TCP 80, TCP 3000) on the VCN's Default
Security List exactly as before — the retry script only automates the
instance launch, not the networking rules, since those are one-time
console clicks that don't depend on capacity timing.

Then continue with the existing `deploy/oracle/setup.sh` flow (rsync or
git-clone the code, run the one-command bootstrap) using the public IP
this script printed.

## Troubleshooting

- **"Config field ... is missing or still a placeholder"** — you left a
  `REPLACE_ME` value in `retry-launch.config.json`. Every field must be
  filled before the script will attempt a launch.
- **"OCI CLI is not authenticated"** — re-run `oci setup config`
  (step 2) and confirm `oci iam region list` works standalone first.
- **It stops immediately citing a limit/quota error (not "capacity")**
  — this usually means the Always Free service limit for A1 OCPUs is
  set to 0 in your tenancy (rare, but happens on some newly created
  accounts pending verification). Console → **Governance &
  Administration → Limits, Quotas and Usage** → search "VM.Standard.A1"
  → confirm the Always Free limit shows 4 OCPUs / 24 GB available. If it
  shows 0, that's an account-provisioning issue to raise with Oracle
  support, not something this script can retry past.
- **Script succeeds but you can't SSH in** — confirm the two ingress
  rules exist (TCP 22 is in the default list already; 80/3000 need
  adding manually per step 5) and that you're using the private key
  matching the `.pub` file referenced in the config.
