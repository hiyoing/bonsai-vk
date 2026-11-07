param([switch]$Kill, [switch]$Cpu)
$ErrorActionPreference = 'SilentlyContinue'

# ABSOLUTE WHITELIST - never touched, matched by exact name OR by any name containing these
$whiteExact = @('Qoder CN','node','python','pythonw','powershell','pwsh','cmd','conhost','explorer',
  'svchost','Registry','csrss','wininit','winlogon','services','lsass','smss','dwm','fontdrvhost',
  'taskhostw','RuntimeBroker','SearchHost','SearchApp','SearchIndexer','StartMenuExperienceHost',
  'ShellExperienceHost','TextInputHost','Widgets','WidgetService','CrossDeviceService','msedgewebview2',
  'RadeonSoftware','AMDRSServ','AMDRSSrcExt','atiesrxx','atiesrz','MSPCManagerService','MsMpEng','NisSrv',
  'HipsDaemon','wsctrl','ZhuDongFangYu','Memory Compression','VulkanRuntime','WmiApRpl','WUDFHost','audiodg')
$whiteSub   = @('qoder','python','node','powershell','cmd','windows','system','radeon','amd','microsoft',
                'defender','hips','antimalware','search','shell','task','svchost','webview')

# HARD targets requested by the user, plus obvious non-dev hogs
# GameViewerServer: user-authorized 2026-11-08 - the remote-viewer server held 1.6 GiB of VRAM and
# pushed dwm to 1.7 GiB, depressing even the pure-CLI control leg to 13.71 tok/s (webui_attrib_v1).
$hard = @('chrome','msedge','javaw','java','steam','EpicGamesLauncher','Discord','Spotify','OneDrive',
          'Telegram','WeChat','WeChatAppEx','QQ','QQEXPLORER','iDingTalk','Feishu','DingTalk','360se',
          '360chrome','hrcoder','ToDesk','SunloginClient','GameViewerServer','aDrive','AliWorkbench','BlueStacks','bsservice')

function Is-White($n) {
  if ($whiteExact -contains $n) { return $true }
  foreach ($w in $whiteSub) { if ($n -like "*$w*") { return $true } }
  return $false
}

$free0 = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB,0)
Write-Output ("AVAILABLE_MB_BEFORE={0}" -f $free0)

if ($Cpu) {
  Write-Output "--- top 15 by CPU across 2 samples ---"
  $a = Get-Process | Select-Object Name,Id,@{n='C';e={$_.CPU}}
  Start-Sleep -Seconds 3
  $b = Get-Process | Select-Object Name,Id,@{n='C';e={$_.CPU}}
  $nc = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
  $rows = @()
  foreach ($x in $b) {
    $y = $a | Where-Object { $_.Id -eq $x.Id } | Select-Object -First 1
    if ($y -and $x.C -and $y.C) {
      $d = ($x.C - $y.C) / 3.0 * 100.0 / $nc
      if ($d -gt 0.5) { $rows += [pscustomobject]@{N=$x.Name;P=$x.Id;Pct=[math]::Round($d,1)} }
    }
  }
  $rows | Sort-Object Pct -Descending | Select-Object -First 15 |
    ForEach-Object { Write-Output ("{0,-26} pid={1,-7} cpu={2,5}% of one core-group" -f $_.N, $_.P, $_.Pct) }
}

Write-Output "--- hard-target scan ---"
$hit = @()
foreach ($p in Get-Process) {
  if ($hard -contains $p.Name) {
    if (Is-White $p.Name) { Write-Output ("  SKIP whitelisted {0} pid={1}" -f $p.Name,$p.Id); continue }
    Write-Output ("  KILL {0,-14} pid={1,-7} ws={2,8:N1} MB" -f $p.Name, $p.Id, ($p.WorkingSet64/1MB))
    $hit += $p
  }
}
if ($hit.Count -eq 0) { Write-Output "  none of the requested kill targets are running" }
if ($Kill -and $hit.Count -gt 0) {
  foreach ($p in $hit) { Stop-Process -Id $p.Id -Force }
  Start-Sleep -Seconds 2
}
Write-Output "--- after ---"
Write-Output ("KILLED={0}" -f $hit.Count)
Write-Output ("AVAILABLE_MB_AFTER={0}" -f [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB,0))
