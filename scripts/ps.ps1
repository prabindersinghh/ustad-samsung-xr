Get-CimInstance Win32_Process -Filter "name='python.exe'" | ForEach-Object {
  $c = $_.CommandLine; if ($c.Length -gt 150) { $c = $c.Substring(0,150) }
  "{0}`t{1} MB`t{2}" -f $_.ProcessId, [int]($_.WorkingSetSize/1MB), $c
}
