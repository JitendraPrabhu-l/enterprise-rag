# Retries an Oracle Cloud A1.Flex instance launch until capacity is available.
#
# Oracle's Always Free ARM (A1.Flex) pool is genuinely, chronically oversubscribed
# in several regions (India South/Hyderabad among them) - "Out of capacity"
# is not a config error, it's the pool being momentarily empty. Retrying the
# same launch request periodically is the standard, Oracle-acknowledged
# workaround: capacity frees up as other tenants' instances terminate, in
# bursts that are impossible to predict from outside.
#
# This script does exactly what re-clicking "Create" in the console does,
# but unattended: read retry-launch.config.json, call `oci compute instance
# launch` with that exact shape/OCPU/memory/image, and on the specific
# "Out of capacity" error, wait and try again. Any OTHER error (bad OCID,
# auth failure, quota exceeded) stops immediately - those will not fix
# themselves by retrying, and don't deserve a background poll loop chewing
# through your API call quota when the config is genuinely wrong.
#
# Usage (from this directory):
#   .\retry-launch.ps1
#   .\retry-launch.ps1 -ConfigPath .\retry-launch.config.json

param(
    [string]$ConfigPath = "$PSScriptRoot\retry-launch.config.json"
)

$ErrorActionPreference = "Stop"

function Write-Status([string]$Message, [string]$Color = "White") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

if (-not (Test-Path $ConfigPath)) {
    Write-Status "Config file not found: $ConfigPath" "Red"
    Write-Status "Copy retry-launch.config.json and fill in your OCIDs first (see SETUP-RETRY-LAUNCH.md)." "Red"
    exit 1
}

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

foreach ($required in @(
    "compartment_id", "availability_domain", "subnet_id", "image_id",
    "ssh_public_key_path", "shape", "ocpus", "memory_in_gbs"
)) {
    $value = $config.$required
    if ([string]::IsNullOrWhiteSpace($value) -or ($value -is [string] -and $value.StartsWith("REPLACE_ME"))) {
        Write-Status "Config field '$required' is missing or still a placeholder - edit $ConfigPath first." "Red"
        exit 1
    }
}

if (-not (Test-Path $config.ssh_public_key_path)) {
    Write-Status "SSH public key not found at: $($config.ssh_public_key_path)" "Red"
    $privateKeyPathHint = $config.ssh_public_key_path -replace '\.pub$', ''
    Write-Status "Generate one with: ssh-keygen -t ed25519 -f $privateKeyPathHint" "Red"
    exit 1
}

# Verify the OCI CLI itself is installed and authenticated before entering
# the retry loop - a broken `oci setup config` would otherwise look
# identical to "still out of capacity" for every single attempt.
try {
    $null = & oci --version 2>&1
} catch {
    Write-Status "OCI CLI not found on PATH. See SETUP-RETRY-LAUNCH.md to install it." "Red"
    exit 1
}

Write-Status "Verifying OCI CLI authentication (oci iam region list)..." "Cyan"
$previousEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$authCheck = & oci iam region list --query "data[0].name" --raw-output 2>&1
$ErrorActionPreference = $previousEap
if ($LASTEXITCODE -ne 0) {
    Write-Status "OCI CLI is not authenticated. Run 'oci setup config' first (see SETUP-RETRY-LAUNCH.md)." "Red"
    Write-Status $authCheck "Red"
    exit 1
}
Write-Status "OCI CLI authenticated OK." "Green"

$sshKey = (Get-Content $config.ssh_public_key_path -Raw).Trim()

# Both --shape-config and --metadata are written to temp JSON files and
# passed as file://... rather than inline JSON strings. PowerShell's native
# argument passing re-quotes/mangles embedded double-quotes when a single
# array element itself contains a JSON object (observed directly: the OCI
# CLI rejected an inline --metadata value with "must be in JSON format"
# even though the JSON was valid) - file:// sidesteps that entirely, and is
# also officially supported by the OCI CLI for exactly this kind of
# complex-type argument.
$shapeConfigFile = New-TemporaryFile
$metadataFile = New-TemporaryFile
try {
    @{ ocpus = $config.ocpus; memoryInGBs = $config.memory_in_gbs } |
        ConvertTo-Json -Compress |
        Set-Content -Path $shapeConfigFile -Encoding Ascii -NoNewline

    @{ ssh_authorized_keys = $sshKey } |
        ConvertTo-Json -Compress |
        Set-Content -Path $metadataFile -Encoding Ascii -NoNewline

    $launchArgs = @(
        "compute", "instance", "launch",
        "--compartment-id", $config.compartment_id,
        "--availability-domain", $config.availability_domain,
        "--shape", $config.shape,
        "--shape-config", "file://$shapeConfigFile",
        "--image-id", $config.image_id,
        "--subnet-id", $config.subnet_id,
        "--display-name", $config.display_name,
        "--assign-public-ip", "true",
        "--boot-volume-size-in-gbs", $config.boot_volume_size_in_gbs,
        "--metadata", "file://$metadataFile",
        "--wait-for-state", "RUNNING"
    )

    $attempt = 0
    $maxAttempts = [int]$config.max_attempts
    $intervalSeconds = [int]$config.retry_interval_seconds

    Write-Status "Starting launch retry loop." "Cyan"
    Write-Status "Shape: $($config.shape) | OCPUs: $($config.ocpus) | Memory: $($config.memory_in_gbs) GB | AD: $($config.availability_domain)" "Cyan"
    Write-Status "Retry interval: ${intervalSeconds}s | Max attempts: $(if ($maxAttempts -le 0) { 'unlimited' } else { $maxAttempts })" "Cyan"
    Write-Status "Press Ctrl+C to stop at any time - no partial resources are left behind on a failed attempt." "Cyan"
    Write-Host ""

    $logPath = "$PSScriptRoot\retry-launch.log"
    Write-Status "Full output of every attempt is also logged to: $logPath" "Cyan"
    Write-Host ""

    while ($true) {
        $attempt++
        Write-Status "Attempt #${attempt}: launching instance..." "White"

        # oci.exe writing to stderr on a non-zero exit must NOT become a
        # PowerShell terminating exception here - under the script-wide
        # $ErrorActionPreference = "Stop", it was doing exactly that
        # (observed directly: the ServiceError/RequestException bodies never
        # reached the capacity-detection logic below, or the log file,
        # because control unwound straight out of this loop via PowerShell's
        # own NativeCommandError before $output was ever assigned). Scoping
        # ErrorActionPreference to Continue for just this call is what makes
        # $output/$LASTEXITCODE actually usable as data afterward.
        $previousEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $output = & oci @launchArgs 2>&1
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousEap

        # Every attempt's full output is appended here regardless of outcome
        # - the terminal itself can wrap/truncate long JSON error bodies,
        # and re-running a one-off repro command by hand (as we did to
        # diagnose the "Out of host capacity" vs RequestException shapes)
        # is exactly the friction this log removes going forward.
        "===== Attempt #${attempt} at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (exit code $exitCode) =====" |
            Out-File -FilePath $logPath -Append -Encoding utf8
        $output | Out-File -FilePath $logPath -Append -Encoding utf8

        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Status "SUCCESS - instance launched and reached RUNNING state." "Green"
            $result = $output | Out-String | ConvertFrom-Json
            $instanceId = $result.data.id
            Write-Status "Instance OCID: $instanceId" "Green"

            Write-Status "Fetching public IP..." "White"
            $previousEap = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $publicIp = & oci compute instance list-vnics --instance-id $instanceId --query 'data[0]."public-ip"' --raw-output 2>&1
            $ErrorActionPreference = $previousEap
            if ($LASTEXITCODE -eq 0) {
                Write-Status "Public IP: $publicIp" "Green"
                $privateKeyPath = $config.ssh_public_key_path -replace '\.pub$', ''
                Write-Status "SSH in with: ssh -i $privateKeyPath ubuntu@$publicIp" "Green"
            } else {
                Write-Status "Instance is running, but IP lookup failed - check the OCI Console for its public IP." "Yellow"
            }
            exit 0
        }

        $outputText = $output | Out-String

        # Oracle returns capacity exhaustion under more than one shape:
        # historically a 400 "LimitExceeded" with "capacity" in the message,
        # but observed directly (2026-07-14, ap-hyderabad-1) as a 500
        # "InternalError" whose message is literally "Out of host capacity."
        # - note "host", not "instance" - so match loosely on "capacity"
        # anywhere in the message rather than pinning to one exact phrase or
        # error code, since Oracle's own classification of this condition is
        # evidently not stable across requests.
        $isCapacityError = $outputText -match "(?i)out of (host )?capacity" -or
                            ($outputText -match "(?i)capacity" -and $outputText -match "(?i)(LimitExceeded|InternalError)")

        # A "RequestException" with a connection/timeout message (observed
        # directly: "The connection to endpoint timed out.") means the
        # request never reached Oracle's API at all - a transient network
        # blip, not a rejection of any kind. Treating this as fatal would
        # kill an unattended multi-hour retry loop over one dropped packet;
        # it gets the same retry treatment as a capacity error, just logged
        # distinctly so the difference is visible in the log/console.
        $isTransientNetworkError = $outputText -match "(?i)RequestException" -and
                                    $outputText -match "(?i)(timed out|timeout|connection (aborted|reset|refused)|temporary failure|name resolution)"

        # Oracle's own API rate limit (observed directly after ~29 launch
        # attempts over ~75 minutes at the fixed 90s interval: "code":
        # "TooManyRequests", HTTP 429). This is Oracle telling us to slow
        # down, not a capacity/config problem - genuinely retryable, but
        # retrying at the SAME interval that triggered the 429 would just
        # trigger it again immediately, so this gets its own much longer
        # backoff (10x the configured interval, floor 600s/10min) instead of
        # the normal retry_interval_seconds.
        $isRateLimited = $outputText -match "(?i)TooManyRequests" -or $outputText -match "(?i)429"

        if (-not ($isCapacityError -or $isTransientNetworkError -or $isRateLimited)) {
            Write-Host ""
            Write-Status "Launch failed with a non-retryable error - stopping (retrying will not fix this)." "Red"
            Write-Status $outputText "Red"
            Write-Status "Full details also saved to: $logPath" "Red"
            exit 1
        }

        if ($isRateLimited) {
            Write-Status "Oracle API rate limit hit (429 TooManyRequests) - backing off longer than usual." "Yellow"
        } elseif ($isTransientNetworkError) {
            Write-Status "Transient network error talking to Oracle's API (connection/timeout, not a rejection) - retrying." "Yellow"
        } else {
            Write-Status "Out of capacity (as expected - this is the known A1.Flex shortage, not a config problem)." "Yellow"
        }

        if ($maxAttempts -gt 0 -and $attempt -ge $maxAttempts) {
            Write-Status "Reached max attempts ($maxAttempts). Stopping." "Yellow"
            exit 2
        }

        $waitSeconds = if ($isRateLimited) { [Math]::Max($intervalSeconds * 10, 600) } else { $intervalSeconds }
        Write-Status "Waiting ${waitSeconds}s before retrying..." "DarkGray"
        Start-Sleep -Seconds $waitSeconds
    }
} finally {
    Remove-Item -Path $shapeConfigFile -ErrorAction SilentlyContinue
    Remove-Item -Path $metadataFile -ErrorAction SilentlyContinue
}
