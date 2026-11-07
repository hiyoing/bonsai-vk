# dl_sdk.ps1 - parallel ranged download of the Vulkan SDK installer.
# The CDN (Cloudflare, HKG edge) caps a single connection at tens of KB/s from this line;
# 8 concurrent byte-range requests is the standard way around a per-connection throttle.
# Total size is not guessed: HEAD reports Content-Length = 303082720 (verified 2026-11-01).

$url  = 'https://sdk.lunarg.com/sdk/download/1.4.363.0/windows/vulkansdk-windows-X64-1.4.363.0.exe'
$total = 303082720
$parts = 8
$dev  = if ($env:DEV) { $env:DEV } else { Join-Path $PSScriptRoot 'toolchain' }
$dir  = Join-Path $dev 'parts'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$size = [math]::Ceiling($total / $parts)
$jobs = @()
for ($i = 0; $i -lt $parts; $i++) {
    $lo = [int64]$i * $size
    if ($lo -ge $total) { break }
    $hi = [math]::Min($lo + $size - 1, $total - 1)
    $out = Join-Path $dir ("part{0:d2}" -f $i)
    if (Test-Path $out) { Remove-Item $out -Force }
    $jobs += Start-Job -ScriptBlock {
        param($u, $range, $o)
        & C:\Windows\System32\curl.exe -sS -L --fail --retry 10 --retry-delay 5 -r $range -o $o $u 2>&1
    } -ArgumentList $url, "$lo-$hi", $out
}
Write-Host "started $($jobs.Count) ranged jobs, $parts x $size B"

$deadline = (Get-Date).AddMinutes(90)
while ($true) {
    $running = @($jobs | Where-Object { $_.State -eq 'Running' })
    $got = (Get-ChildItem $dir -Filter 'part*' -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    Write-Host ("{0:HH:mm:ss} running={1} got={2:N1}/{3:N1} MB ({4:N1}%)" -f (Get-Date), $running.Count, ($got/1MB), ($total/1MB), (100*$got/$total))
    if ($running.Count -eq 0) { break }
    if ((Get-Date) -gt $deadline) { Write-Host 'DEADLINE'; $running | Stop-Job; break }
    Start-Sleep -Seconds 30
}
$jobs | Remove-Job -Force

# concatenate and verify against the SHA256 that winget's manifest pins
$dst = Join-Path $dev 'installer\vulkansdk-windows-X64-1.4.363.0.exe'
New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
if (Test-Path $dst) { Remove-Item $dst -Force }
$fs = [System.IO.File]::Create($dst)
try {
    Get-ChildItem $dir -Filter 'part*' | Sort-Object Name | ForEach-Object {
        $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
        $fs.Write($bytes, 0, $bytes.Length)
    }
} finally { $fs.Close() }

$len = (Get-Item $dst).Length
$sha = (Get-FileHash $dst -Algorithm SHA256).Hash.ToLower()
Write-Host "size=$len expect=$total"
Write-Host "sha256=$sha"
Write-Host "expect  =94a82d378f7a5e3e54c9db7d2fb7016af136e14ac0a18dbf0f2f67a36352d141"
if ($len -eq $total -and $sha -eq '94a82d378f7a5e3e54c9db7d2fb7016af136e14ac0a18dbf0f2f67a36352d141') {
    Write-Host 'SDK_INSTALLER_OK'
} else {
    Write-Host 'SDK_INSTALLER_BAD'
}
