param(
    [Parameter(Mandatory=$true)][string]$Name,      # process name substring, e.g. llama-cli
    [string]$Out = "vram.tsv",
    [double]$Seconds = 900.0
)
# Samples the WDDM per-process GPU memory counters (the same ones GPU-aware Task Manager
# shows) for the target process, plus system free memory and physical-disk read throughput,
# so "mechanical disk -> VRAM" is backed by a counter instead of an assumption.
# Get-Counter's minimum -SampleInterval is 1 s; one call per tick reads every path at once.
# Columns: t_ms  local_MB  nonlocal_MB  shared_MB  disk_read_MBps  avail_MB  pids
$ErrorActionPreference = 'SilentlyContinue'
$paths = @('\GPU Process Memory(*)\Local Usage',
           '\GPU Process Memory(*)\Non Local Usage',
           '\GPU Process Memory(*)\Shared Usage',
           '\PhysicalDisk(_Total)\Disk Read Bytes/sec',
           '\Memory\Available MBytes')
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$fs = New-Object System.IO.StreamWriter($Out, $false)
$fs.WriteLine("t_ms`tlocal_MB`tnonlocal_MB`tshared_MB`tdisk_read_MBps`tavail_MB`tpids")
$seen = @()
while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
    $t0 = $sw.Elapsed.TotalMilliseconds
    $pids = @()
    foreach ($p in (Get-Process)) {
        if ($p.ProcessName -like "*$Name*") { $pids += [string]$p.Id; if ($seen -notcontains $p.Id) { $seen += $p.Id } }
    }
    $r = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1
    $l = 0.0; $n = 0.0; $s = 0.0; $d = 0.0; $av = 0.0
    if ($r) {
        foreach ($c in $r.CounterSamples) {
            $p2 = $c.Path.ToLower()
            if ($p2 -like '*gpu process memory*') {
                $m = [regex]::Match($c.InstanceName, 'pid_(\d+)_')
                if ($m.Success -and ($pids -contains $m.Groups[1].Value)) {
                    if ($p2 -like '*non local*') { $n += $c.CookedValue }
                    elseif ($p2 -like '*local usage*') { $l += $c.CookedValue }
                    elseif ($p2 -like '*shared*') { $s += $c.CookedValue }
                }
            }
            elseif ($p2 -like '*disk read bytes*') { $d = $c.CookedValue }
            elseif ($p2 -like '*available mbytes*') { $av = $c.CookedValue }
        }
    }
    $fs.WriteLine(("{0}`t{1:N1}`t{2:N1}`t{3:N1}`t{4:N1}`t{5:N0}`t{6}" -f [math]::Round($t0,1), ($l/1MB), ($n/1MB), ($s/1MB), ($d/1MB), $av, ($pids -join '/')))
    $fs.Flush()
}
$fs.Close()
Write-Output ("PIDS_SEEN: " + ($seen -join ','))
Write-Output ("SAMPLED_S: " + [math]::Round($sw.Elapsed.TotalSeconds, 1))
